import numpy as np

from app.errors import LivenessFailedError


class FakeLivenessChecker:
    """Test double: passes by default; set force_spoof=True to fail ensure_live."""

    def __init__(self, *, force_spoof: bool = False, score: float = 0.95) -> None:
        self.force_spoof = force_spoof
        self._score = score

    def ready(self) -> bool:
        return True

    def score_bgr(self, image_bgr: np.ndarray) -> float:
        if self.force_spoof:
            return 0.05
        return float(self._score)

    def ensure_live(self, image_bgr: np.ndarray, *, threshold: float) -> float:
        score = self.score_bgr(image_bgr)
        if score < threshold:
            raise LivenessFailedError(
                f"liveness check failed (score={score:.4f} < {threshold:.4f})"
            )
        return score


class UnavailableLivenessChecker:
    """Used when the ONNX liveness model is not installed."""

    def ready(self) -> bool:
        return False

    def score_bgr(self, image_bgr: np.ndarray) -> float:
        raise RuntimeError("liveness model not available")

    def ensure_live(self, image_bgr: np.ndarray, *, threshold: float) -> float:
        raise RuntimeError("liveness model not available")
