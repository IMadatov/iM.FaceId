from dataclasses import dataclass

from app.domain.checks.base import FrameSequence
from app.domain.errors import DependencyUnavailableError, LivenessFailedError

NO_KEYPOINTS = (
    "planarity check needs facial keypoints, which this model pack "
    "does not provide (set LIVENESS_MIN_PLANARITY=0 to disable)"
)


@dataclass(frozen=True)
class InterFrameDistanceCheck:
    """Reject a sequence of near-identical frames (a still image replayed)."""

    min_distance: float
    name: str = "inter_frame_distance"

    def verify(self, sequence: FrameSequence) -> None:
        max_distance = sequence.max_pairwise_distance
        if max_distance < self.min_distance:
            raise LivenessFailedError(
                f"liveness check failed (frames too similar: "
                f"distance={max_distance:.4f} < {self.min_distance:.4f})"
            )


@dataclass(frozen=True)
class KeypointsAvailableCheck:
    """The planarity pair needs landmarks from every frame."""

    name: str = "keypoints_available"

    def verify(self, sequence: FrameSequence) -> None:
        if any(kps is None for kps in sequence.keypoints):
            raise DependencyUnavailableError(NO_KEYPOINTS)


@dataclass(frozen=True)
class ViewChangeCheck:
    """Require enough head turn for the parallax score to carry information."""

    min_view_change: float
    name: str = "view_change"

    def verify(self, sequence: FrameSequence) -> None:
        spread = sequence.view_change_spread
        if spread < self.min_view_change:
            raise LivenessFailedError(
                f"liveness check failed (not enough head movement: "
                f"view change={spread:.4f} < {self.min_view_change:.4f})"
            )


@dataclass(frozen=True)
class PlanarityCheck:
    """Reject a flat replay: a real nose shows motion parallax, paper does not."""

    min_planarity: float
    name: str = "planarity"

    def verify(self, sequence: FrameSequence) -> None:
        residual = sequence.planarity_residual
        if residual < self.min_planarity:
            raise LivenessFailedError(
                f"liveness check failed (face looks flat: "
                f"parallax={residual:.4f} < {self.min_planarity:.4f})"
            )
