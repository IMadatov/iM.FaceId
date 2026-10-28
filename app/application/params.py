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
