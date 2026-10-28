from app.domain.checks.base import FrameCheck, SequenceCheck
from app.domain.checks.liveness import (
    LivenessReadyCheck,
    LivenessScoreCheck,
    MeanLivenessCheck,
)
from app.domain.checks.motion import (
    InterFrameDistanceCheck,
    KeypointsAvailableCheck,
    PlanarityCheck,
    ViewChangeCheck,
)
from app.domain.checks.quality import (
    BlurCheck,
    BrightnessMaxCheck,
    BrightnessMinCheck,
    FaceCropCheck,
    MinSizeCheck,
)


def build_quality_chain(gate) -> list[FrameCheck]:
    """Image-quality checks in the legacy ensure_face_quality order.

    `gate` is an application.params.GateParams. Typed loosely to keep the
    domain layer from importing the application layer; only attribute reads
    happen here.
    """
    chain: list[FrameCheck] = []
    if gate.min_side_px > 0:
        chain.append(MinSizeCheck(min_side_px=gate.min_side_px))
    chain.append(FaceCropCheck())  # always: the legacy gate cropped unconditionally
    if gate.blur_min_variance > 0:
        chain.append(BlurCheck(min_variance=gate.blur_min_variance))
    if gate.brightness_min > 0:
        chain.append(BrightnessMinCheck(minimum=gate.brightness_min))
    if gate.brightness_max < 255:
        chain.append(BrightnessMaxCheck(maximum=gate.brightness_max))
    return chain


def build_single_frame_chain(gate) -> list[FrameCheck]:
    """Quality, then the model-ready probe, then the single-frame threshold."""
    return [
        *build_quality_chain(gate),
        LivenessReadyCheck(),
        LivenessScoreCheck(threshold=gate.liveness_threshold),
    ]


def planarity_enabled(params) -> bool:
    """True when the response must report a planarity_residual number.

    The legacy early return needed BOTH thresholds at or below zero, so a
    config with min_planarity == 0 and min_view_change > 0 still measures
    and reports the residual.
    """
    return params.min_planarity > 0.0 or params.min_view_change > 0.0


def build_sequence_chain(params) -> list[SequenceCheck]:
    """Multi-frame checks in the legacy _run_multiframe_liveness order.

    `params` is an application.params.MultiframeParams.
    """
    chain: list[SequenceCheck] = [
        MeanLivenessCheck(threshold=params.gate.liveness_threshold),
        InterFrameDistanceCheck(min_distance=params.min_distance),
    ]
    if planarity_enabled(params):
        chain.append(KeypointsAvailableCheck())
        chain.append(ViewChangeCheck(min_view_change=params.min_view_change))
        chain.append(PlanarityCheck(min_planarity=params.min_planarity))
    return chain
