from dataclasses import dataclass


@dataclass(frozen=True)
class EnrollResult:
    face_id: str
    liveness_score: float | None


@dataclass(frozen=True)
class CheckResult:
    liveness_score: float | None


@dataclass(frozen=True)
class UpdateResult:
    face_id: str
    liveness_score: float | None


@dataclass(frozen=True)
class VerifyResult:
    face_id: str
    match: bool
    score: float
    threshold: float
    liveness_score: float | None


@dataclass(frozen=True)
class SearchResult:
    hits: list[tuple[str, float]]
    threshold: float
    liveness_score: float | None


@dataclass(frozen=True)
class LivenessResult:
    mean_score: float
    frame_count: int
    inter_frame_distance: float
    planarity_residual: float | None
