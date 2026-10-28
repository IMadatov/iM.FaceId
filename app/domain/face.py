from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DetectedFace:
    embedding: np.ndarray
    bbox: np.ndarray  # [x1, y1, x2, y2]
    # (5, 2) eyes/nose/mouth-corner landmarks; None when the model pack has no
    # keypoint head. Only the planarity liveness check needs them.
    kps: np.ndarray | None = None
