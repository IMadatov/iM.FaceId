from dataclasses import dataclass


@dataclass(frozen=True)
class GateParams:
    """Resolved single-frame thresholds. Built by the API layer from Settings
    plus per-request overrides, so the application layer never sees Settings."""

    blur_min_variance: float
    brightness_min: float
    brightness_max: float
    min_side_px: float
    liveness_threshold: float


@dataclass(frozen=True)
class MultiframeParams:
    """Resolved multi-frame thresholds."""

    gate: GateParams
    max_frames: int
    min_distance: float
    min_planarity: float
    min_view_change: float


@dataclass(frozen=True)
class ImageLimits:
    """Decode guards. Mirrors Settings.max_image_side / max_upload_bytes."""

    max_side: int
    max_bytes: int


@dataclass(frozen=True)
class SingleFrameRequest:
    """One image plus every threshold already resolved against Settings."""

    image_base64: str
    liveness: bool
    limits: ImageLimits
    gate: GateParams


@dataclass(frozen=True)
class VerifyRequest(SingleFrameRequest):
    match_threshold: float


@dataclass(frozen=True)
class SearchRequest(SingleFrameRequest):
    limit: int
    min_score: float
