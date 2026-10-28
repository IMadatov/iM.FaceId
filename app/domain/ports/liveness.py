from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class LivenessScorer(Protocol):
    def ready(self) -> bool: ...

    # bbox is positional-or-keyword to match OnnxLivenessChecker.score_bgr.
    def score_bgr(self, image_bgr: np.ndarray, bbox: np.ndarray | None = None) -> float: ...
