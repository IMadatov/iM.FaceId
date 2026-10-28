from dataclasses import dataclass
from functools import cached_property
from typing import Protocol, runtime_checkable

import numpy as np

from app.domain.face import DetectedFace
from app.domain.matching.cosine import cosine_distance
from app.domain.planarity import max_planarity_residual, view_change_spread
from app.domain.ports.liveness import LivenessScorer
from app.domain.ports.metrics import FaceImageMetrics


# Not frozen: cached_property needs a __dict__ to memoise into. Treat every
# field as read-only by convention — nothing in the codebase reassigns them.
@dataclass
class Frame:
    """One decoded image plus its detection, with measurements computed once."""

    image_bgr: np.ndarray
    detected: DetectedFace
    metrics: FaceImageMetrics
    scorer: LivenessScorer | None = None

    @cached_property
    def min_side(self) -> float:
        x1, y1, x2, y2 = (float(v) for v in self.detected.bbox[:4])
        return min(max(0.0, x2 - x1), max(0.0, y2 - y1))

    @cached_property
    def crop(self) -> np.ndarray:
        return self.metrics.crop(self.image_bgr, self.detected.bbox)

    @cached_property
    def gray(self) -> np.ndarray:
        return self.metrics.gray(self.crop)

    @cached_property
    def liveness_score(self) -> float:
        if self.scorer is None:  # pragma: no cover - wiring error, not an input
            raise RuntimeError("Frame.liveness_score requires a scorer")
        return self.scorer.score_bgr(self.image_bgr, self.detected.bbox)


@dataclass
class FrameSequence:
    """Several frames judged together. Metrics are computed once and shared."""

    frames: list[Frame]

    @cached_property
    def liveness_scores(self) -> list[float]:
        return [f.liveness_score for f in self.frames]

    @cached_property
    def mean_liveness(self) -> float:
        scores = self.liveness_scores
        return float(sum(scores) / len(scores))

    @cached_property
    def max_pairwise_distance(self) -> float:
        embeddings = [f.detected.embedding for f in self.frames]
        return max(
            cosine_distance(embeddings[i], embeddings[j])
            for i in range(len(embeddings))
            for j in range(i + 1, len(embeddings))
        )

    @cached_property
    def keypoints(self) -> list[np.ndarray | None]:
        return [f.detected.kps for f in self.frames]

    @cached_property
    def view_change_spread(self) -> float:
        return view_change_spread(self.keypoints)

    @cached_property
    def planarity_residual(self) -> float:
        return max_planarity_residual(self.keypoints)


@runtime_checkable
class FrameCheck(Protocol):
    """One rule about one frame. Raises a domain error, or returns None."""

    name: str

    def verify(self, frame: Frame) -> None: ...


@runtime_checkable
class SequenceCheck(Protocol):
    """One rule about a whole frame sequence."""

    name: str

    def verify(self, sequence: FrameSequence) -> None: ...
