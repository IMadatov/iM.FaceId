from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class DetectedFace:
    embedding: np.ndarray
    bbox: np.ndarray  # [x1, y1, x2, y2]
    # (5, 2) eyes/nose/mouth-corner landmarks; None when the model pack has no
    # keypoint head. Only the planarity liveness check needs them.
    kps: np.ndarray | None = None


class FacePipeline(Protocol):
    def detect_bgr(self, image_bgr: np.ndarray) -> DetectedFace: ...
    def embed_bgr(self, image_bgr: np.ndarray) -> np.ndarray: ...
    def ready(self) -> bool: ...
