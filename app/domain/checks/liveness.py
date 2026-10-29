from dataclasses import dataclass

from app.domain.checks.base import Frame, FrameSequence
from app.domain.errors import DependencyUnavailableError, LivenessFailedError

MODEL_UNAVAILABLE = (
    "liveness model not available "
    "(run scripts/download_liveness_model.py)"
)


@dataclass(frozen=True)
class LivenessReadyCheck:
    """Probe the scorer before touching it.

    UnavailableLivenessChecker.score_bgr raises RuntimeError, which would
    surface as a 500; this converts the condition into a 503 with the
    legacy detail string.
    """

    name: str = "liveness_ready"

    def verify(self, frame: Frame) -> None:
        if frame.scorer is None or not frame.scorer.ready():
            raise DependencyUnavailableError(MODEL_UNAVAILABLE)


@dataclass(frozen=True)
class LivenessScoreCheck:
    """Single-frame anti-spoof threshold."""

    threshold: float
    name: str = "liveness_score"

    def verify(self, frame: Frame) -> None:
        score = frame.liveness_score
        if score < self.threshold:
            raise LivenessFailedError(
                f"liveness check failed (score={score:.4f} < {self.threshold:.4f})"
            )


@dataclass(frozen=True)
class MeanLivenessCheck:
    """Multi-frame anti-spoof threshold on the mean score."""

    threshold: float
    name: str = "mean_liveness"

    def verify(self, sequence: FrameSequence) -> None:
        mean_score = sequence.mean_liveness
        if mean_score < self.threshold:
            raise LivenessFailedError(
                f"liveness check failed (mean score={mean_score:.4f} "
                f"< {self.threshold:.4f})"
            )
