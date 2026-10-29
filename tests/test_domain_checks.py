import numpy as np
import pytest

from app.application.params import GateParams, MultiframeParams
from app.domain.checks.base import Frame, FrameSequence
from app.domain.checks.chains import (
    build_quality_chain,
    build_single_frame_chain,
    build_sequence_chain,
    planarity_enabled,
)
from app.domain.errors import DependencyUnavailableError, ImageQualityError, LivenessFailedError
from app.domain.face import DetectedFace
from app.infrastructure.pipeline.metrics import OpenCvFaceImageMetrics

METRICS = OpenCvFaceImageMetrics()


def _gate(**over) -> GateParams:
    base = dict(
        blur_min_variance=0.0,
        brightness_min=0.0,
        brightness_max=255.0,
        min_side_px=0.0,
        liveness_threshold=0.5,
    )
    base.update(over)
    return GateParams(**base)


def _frame(image_bgr: np.ndarray, bbox, kps=None, scorer=None) -> Frame:
    detected = DetectedFace(
        embedding=np.ones(512, dtype=np.float32),
        bbox=np.asarray(bbox, dtype=np.float32),
        kps=kps,
    )
    return Frame(image_bgr=image_bgr, detected=detected, metrics=METRICS, scorer=scorer)


def _run(chain, frame):
    for check in chain:
        check.verify(frame)


# --- Review Focus #2 -------------------------------------------------------
def test_face_crop_check_runs_with_all_thresholds_relaxed():
    """Empty crop is rejected even when every threshold is relaxed to off.

    `_crop_face_bgr`'s clamping (`ix2 = max(ix1 + 1, ...)`) means a real
    OpenCvFaceImageMetrics crop can never actually be zero-area for any
    bbox/image combination -- the same is true of the legacy
    `ensure_face_quality`, which used the identical arithmetic. So this
    uses a fake metrics object that reports an empty crop directly, to pin
    FaceCropCheck's own behaviour independent of that (pre-existing, both
    old and new) dead branch.
    """

    class _EmptyCropMetrics:
        def crop(self, image_bgr, bbox):
            return np.zeros((0, 0, 3), dtype=np.uint8)

        def gray(self, crop_bgr):
            raise AssertionError("must not be reached: crop check runs first")

        def laplacian_variance(self, gray):
            raise AssertionError("must not be reached: crop check runs first")

        def mean_brightness(self, gray):
            raise AssertionError("must not be reached: crop check runs first")

    image = np.full((100, 100, 3), 128, dtype=np.uint8)
    detected = DetectedFace(
        embedding=np.ones(512, dtype=np.float32),
        bbox=np.asarray([10.0, 10.0, 90.0, 90.0], dtype=np.float32),
        kps=None,
    )
    frame = Frame(image_bgr=image, detected=detected, metrics=_EmptyCropMetrics())
    chain = build_quality_chain(_gate())
    with pytest.raises(ImageQualityError, match=r"^image quality failed: empty face crop$"):
        _run(chain, frame)


def test_min_size_check_message_is_verbatim():
    image = np.full((100, 100, 3), 128, dtype=np.uint8)
    frame = _frame(image, [0.0, 0.0, 20.0, 20.0])
    chain = build_quality_chain(_gate(min_side_px=80.0))
    with pytest.raises(ImageQualityError) as exc:
        _run(chain, frame)
    assert str(exc.value) == (
        "image quality failed: face too small (min side=20.0px < 80.0px)"
    )


def test_min_size_check_absent_when_threshold_is_zero():
    names = [c.name for c in build_quality_chain(_gate(min_side_px=0.0))]
    assert "min_side" not in names
    assert "face_crop" in names


def test_blur_check_rejects_flat_crop():
    image = np.full((100, 100, 3), 128, dtype=np.uint8)  # laplacian variance == 0
    frame = _frame(image, [10.0, 10.0, 90.0, 90.0])
    chain = build_quality_chain(_gate(blur_min_variance=50.0))
    with pytest.raises(ImageQualityError) as exc:
        _run(chain, frame)
    assert str(exc.value) == (
        "image quality failed: blur too high (laplacian_var=0.00 < 50.00)"
    )


def test_brightness_min_check_message_is_verbatim():
    image = np.full((100, 100, 3), 10, dtype=np.uint8)
    frame = _frame(image, [10.0, 10.0, 90.0, 90.0])
    chain = build_quality_chain(_gate(brightness_min=40.0))
    with pytest.raises(ImageQualityError) as exc:
        _run(chain, frame)
    assert str(exc.value) == (
        "image quality failed: brightness too low (mean=10.0 < 40.0)"
    )


def test_brightness_max_check_message_is_verbatim():
    image = np.full((100, 100, 3), 240, dtype=np.uint8)
    frame = _frame(image, [10.0, 10.0, 90.0, 90.0])
    chain = build_quality_chain(_gate(brightness_max=220.0))
    with pytest.raises(ImageQualityError) as exc:
        _run(chain, frame)
    assert str(exc.value) == (
        "image quality failed: brightness too high (mean=240.0 > 220.0)"
    )


def test_quality_chain_order_matches_legacy_gate():
    chain = build_quality_chain(
        _gate(min_side_px=80.0, blur_min_variance=50.0,
              brightness_min=40.0, brightness_max=220.0)
    )
    assert [c.name for c in chain] == [
        "min_side", "face_crop", "blur", "brightness_min", "brightness_max",
    ]


def _mf(**over) -> MultiframeParams:
    gate_over = {k: over.pop(k) for k in list(over) if k in GateParams.__annotations__}
    base = dict(max_frames=5, min_distance=0.05, min_planarity=0.06, min_view_change=0.025)
    base.update(over)
    return MultiframeParams(gate=_gate(**gate_over), **base)


# --- Review Focus #3 -------------------------------------------------------
def test_planarity_residual_reported_when_only_view_change_enabled():
    """min_planarity == 0 does NOT disable the measurement; min_view_change > 0 keeps it on."""
    params = _mf(min_planarity=0.0, min_view_change=0.025)
    assert planarity_enabled(params) is True
    names = [c.name for c in build_sequence_chain(params)]
    assert names == [
        "mean_liveness", "inter_frame_distance",
        "keypoints_available", "view_change", "planarity",
    ]


def test_planarity_disabled_only_when_both_thresholds_are_zero():
    params = _mf(min_planarity=0.0, min_view_change=0.0)
    assert planarity_enabled(params) is False
    assert [c.name for c in build_sequence_chain(params)] == [
        "mean_liveness", "inter_frame_distance",
    ]


def test_keypoints_check_message_is_verbatim():
    from app.domain.checks.motion import KeypointsAvailableCheck

    image = np.full((100, 100, 3), 128, dtype=np.uint8)
    sequence = FrameSequence(frames=[
        _frame(image, [0.0, 0.0, 90.0, 90.0], kps=None),
        _frame(image, [0.0, 0.0, 90.0, 90.0], kps=None),
    ])
    with pytest.raises(DependencyUnavailableError) as exc:
        KeypointsAvailableCheck().verify(sequence)
    assert str(exc.value) == (
        "planarity check needs facial keypoints, which this model pack "
        "does not provide (set LIVENESS_MIN_PLANARITY=0 to disable)"
    )


def test_liveness_ready_check_rejects_unavailable_checker():
    from app.domain.checks.liveness import LivenessReadyCheck
    from app.infrastructure.pipeline.fake_liveness import UnavailableLivenessChecker

    image = np.full((100, 100, 3), 128, dtype=np.uint8)
    frame = _frame(image, [0.0, 0.0, 90.0, 90.0], scorer=UnavailableLivenessChecker())
    with pytest.raises(DependencyUnavailableError) as exc:
        LivenessReadyCheck().verify(frame)
    assert str(exc.value) == (
        "liveness model not available (run scripts/download_liveness_model.py)"
    )


def test_single_frame_liveness_threshold_message_is_verbatim():
    from app.domain.checks.liveness import LivenessScoreCheck
    from app.infrastructure.pipeline.fake_liveness import FakeLivenessChecker

    image = np.full((100, 100, 3), 128, dtype=np.uint8)
    frame = _frame(image, [0.0, 0.0, 90.0, 90.0],
                   scorer=FakeLivenessChecker(score=0.10))
    with pytest.raises(LivenessFailedError) as exc:
        LivenessScoreCheck(threshold=0.5).verify(frame)
    assert str(exc.value) == "liveness check failed (score=0.1000 < 0.5000)"


def test_liveness_score_is_computed_once_per_frame():
    class CountingScorer:
        def __init__(self) -> None:
            self.calls = 0

        def ready(self) -> bool:
            return True

        def score_bgr(self, image_bgr, bbox=None) -> float:
            self.calls += 1
            return 0.9

    image = np.full((100, 100, 3), 128, dtype=np.uint8)
    scorer = CountingScorer()
    frame = _frame(image, [0.0, 0.0, 90.0, 90.0], scorer=scorer)
    for check in build_single_frame_chain(_gate()):
        check.verify(frame)
    assert frame.liveness_score == pytest.approx(0.9)
    assert scorer.calls == 1, "cached_property must not re-run inference"
