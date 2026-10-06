import uuid
from collections.abc import Callable
from typing import TypeVar

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.config import Settings
from app.errors import DependencyUnavailableError

T = TypeVar("T")

_UNAVAILABLE_ERRORS = (
    ResponseHandlingException,
    UnexpectedResponse,
    ConnectionError,
    TimeoutError,
    OSError,
)


def _is_valid_id(face_id: str) -> bool:
    try:
        uuid.UUID(face_id)
    except (ValueError, AttributeError, TypeError):
        return False
    return True


class QdrantFaceStore:
    def __init__(self, client: QdrantClient, collection: str, dim: int) -> None:
        self._client = client
        self._collection = collection
        self._dim = dim
        self.ensure_collection()

    @classmethod
    def from_settings(cls, settings: Settings) -> "QdrantFaceStore":
        return cls(
            client=QdrantClient(url=settings.qdrant_url),
            collection=settings.qdrant_collection,
            dim=settings.embedding_dim,
        )

    def _call(self, fn: Callable[[], T]) -> T:
        try:
            return fn()
        except DependencyUnavailableError:
            raise
        except _UNAVAILABLE_ERRORS as exc:
            raise DependencyUnavailableError(f"Qdrant unavailable: {exc}") from exc

    def ensure_collection(self) -> None:
        def _ensure() -> None:
            if not self._client.collection_exists(self._collection):
                self._client.create_collection(
                    collection_name=self._collection,
                    vectors_config=VectorParams(
                        size=self._dim, distance=Distance.COSINE
                    ),
                )

        self._call(_ensure)

    def upsert(self, face_id: str, vector: np.ndarray) -> None:
        if not _is_valid_id(face_id):
            raise ValueError(f"face_id must be a UUID: {face_id!r}")
        vec = vector.astype(np.float32, copy=False).reshape(-1).tolist()
        point = PointStruct(id=face_id, vector=vec, payload={"face_id": face_id})
        self._call(
            lambda: self._client.upsert(
                collection_name=self._collection, points=[point]
            )
        )

    def _retrieve(self, face_id: str, with_vectors: bool) -> list:
        return self._call(
            lambda: self._client.retrieve(
                collection_name=self._collection,
                ids=[face_id],
                with_vectors=with_vectors,
                with_payload=False,
            )
        )

    def get(self, face_id: str) -> np.ndarray | None:
        if not _is_valid_id(face_id):
            return None
        records = self._retrieve(face_id, with_vectors=True)
        if not records:
            return None
        return np.asarray(records[0].vector, dtype=np.float32)

    def delete(self, face_id: str) -> bool:
        if not _is_valid_id(face_id):
            return False
        if not self._retrieve(face_id, with_vectors=False):
            return False
        self._call(
            lambda: self._client.delete(
                collection_name=self._collection, points_selector=[face_id]
            )
        )
        return True

    def clear(self) -> int:
        def _count() -> int:
            return int(self._client.count(collection_name=self._collection).count)

        n = self._call(_count)

        def _wipe() -> None:
            if self._client.collection_exists(self._collection):
                self._client.delete_collection(collection_name=self._collection)

        self._call(_wipe)
        self.ensure_collection()
        return n

    def count(self) -> int:
        def _count() -> int:
            return int(self._client.count(collection_name=self._collection).count)

        return self._call(_count)

    def list_ids(self, *, limit: int, offset: int) -> list[str]:
        if limit < 1 or offset < 0:
            return []
        ids: list[str] = []
        skipped = 0
        next_offset = None

        def _scroll(scroll_offset):
            return self._client.scroll(
                collection_name=self._collection,
                limit=256,
                offset=scroll_offset,
                with_vectors=False,
                with_payload=False,
            )

        while len(ids) < limit:
            scroll_offset = next_offset
            records, next_offset = self._call(lambda o=scroll_offset: _scroll(o))
            for rec in records:
                if skipped < offset:
                    skipped += 1
                    continue
                ids.append(str(rec.id))
                if len(ids) >= limit:
                    break
            if next_offset is None:
                break
        return ids

    def search(
        self, vector: np.ndarray, *, limit: int, min_score: float
    ) -> list[tuple[str, float]]:
        query = vector.astype(np.float32, copy=False).reshape(-1).tolist()
        response = self._call(
            lambda: self._client.query_points(
                collection_name=self._collection,
                query=query,
                limit=limit,
                score_threshold=min_score,
            )
        )
        return [(str(p.id), float(p.score)) for p in response.points]

    def list_all(self) -> list[tuple[str, np.ndarray]]:
        """Scroll all points with vectors from the collection."""
        out: list[tuple[str, np.ndarray]] = []
        next_offset = None

        def _scroll(offset):
            return self._client.scroll(
                collection_name=self._collection,
                limit=256,
                offset=offset,
                with_vectors=True,
                with_payload=False,
            )

        while True:
            offset = next_offset
            records, next_offset = self._call(lambda o=offset: _scroll(o))
            for rec in records:
                out.append((str(rec.id), np.asarray(rec.vector, dtype=np.float32)))
            if next_offset is None:
                break
        return out

    def ping(self) -> bool:
        try:
            self._client.get_collections()
            return True
        except Exception:
            return False
