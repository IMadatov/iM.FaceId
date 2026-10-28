from app.domain.face import DetectedFace
from typing import Protocol

import numpy as np


class FacePipeline(Protocol):
    def detect_bgr(self, image_bgr: np.ndarray) -> DetectedFace: ...
    def embed_bgr(self, image_bgr: np.ndarray) -> np.ndarray: ...
    def ready(self) -> bool: ...


__all__ = ["DetectedFace", "FacePipeline"]
