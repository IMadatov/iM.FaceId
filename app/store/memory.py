import numpy as np

from app.matching.cosine import cosine_similarity


class MemoryFaceStore:
    def __init__(self) -> None:
        self._data: dict[str, np.ndarray] = {}

    def upsert(self, face_id: str, vector: np.ndarray) -> None:
        self._data[face_id] = vector.astype(np.float32, copy=True).reshape(-1)

    def get(self, face_id: str) -> np.ndarray | None:
        v = self._data.get(face_id)
        return None if v is None else v.copy()

    def delete(self, face_id: str) -> bool:
        return self._data.pop(face_id, None) is not None

    def search(
        self, vector: np.ndarray, *, limit: int, min_score: float
    ) -> list[tuple[str, float]]:
        scored = [
            (fid, cosine_similarity(vector, vec))
            for fid, vec in self._data.items()
        ]
        scored = [x for x in scored if x[1] >= min_score]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def list_all(self) -> list[tuple[str, np.ndarray]]:
        return [(fid, vec.copy()) for fid, vec in self._data.items()]

    def ping(self) -> bool:
        return True
