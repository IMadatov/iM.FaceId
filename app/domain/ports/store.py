from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class FaceReader(Protocol):
    def get(self, face_id: str) -> np.ndarray | None: ...


@runtime_checkable
class FaceWriter(Protocol):
    def upsert(self, face_id: str, vector: np.ndarray) -> None: ...
    def delete(self, face_id: str) -> bool: ...


@runtime_checkable
class FaceSearcher(Protocol):
    def search(
        self, vector: np.ndarray, *, limit: int, min_score: float
    ) -> list[tuple[str, float]]: ...


@runtime_checkable
class FaceCatalog(Protocol):
    def count(self) -> int: ...
    def list_ids(self, *, limit: int, offset: int) -> list[str]: ...
    def list_all(self) -> list[tuple[str, np.ndarray]]: ...
    def clear(self) -> int: ...


@runtime_checkable
class HealthProbe(Protocol):
    def ping(self) -> bool: ...


@runtime_checkable
class FaceStore(FaceReader, FaceWriter, FaceSearcher, FaceCatalog, HealthProbe, Protocol):
    """The whole surface. Only the composition root and health endpoints need it."""
