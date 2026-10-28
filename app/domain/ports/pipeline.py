from typing import Protocol, runtime_checkable

import numpy as np

from app.domain.face import DetectedFace


@runtime_checkable
class FaceDetector(Protocol):
    def detect_bgr(self, image_bgr: np.ndarray) -> DetectedFace: ...


@runtime_checkable
class FaceEmbedder(Protocol):
    def embed_bgr(self, image_bgr: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class PipelineProbe(Protocol):
    def ready(self) -> bool: ...


@runtime_checkable
class FacePipeline(FaceDetector, FaceEmbedder, PipelineProbe, Protocol):
    """The whole pipeline surface, for the composition root and /stats."""
