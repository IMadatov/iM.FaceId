import base64
import inspect

import numpy as np
import pytest

from app.application.gate import FaceGate
from app.application.image_loader import Base64ImageLoader
from app.application.params import GateParams, ImageLimits, SingleFrameRequest
from app.application.use_cases.check_face import CheckFaceUseCase
from app.application.use_cases.enroll_face import EnrollFaceUseCase
from app.domain.errors import DependencyUnavailableError, InvalidImageError
from app.domain.face import DetectedFace
from app.infrastructure.imaging import OpenCvImageDecoder
from app.infrastructure.pipeline.fake_liveness import FakeLivenessChecker, UnavailableLivenessChecker
from app.infrastructure.pipeline.metrics import OpenCvFaceImageMetrics

LIMITS = ImageLimits(max_side=640, max_bytes=5_000_000)
STRICT_GATE = GateParams(
    blur_min_variance=50.0,
    brightness_min=40.0,
    brightness_max=220.0,
    min_side_px=80.0,
    liveness_threshold=0.5,
)


def _png_base64() -> str:
    """A 200x200 mid-grey PNG: decodes fine, fails every quality gate."""
    import cv2

    image = np.full((200, 200, 3), 128, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", image)
    assert ok
    return base64.b64encode(buf.tobytes()).decode()


class StubDetector:
    def __init__(self, bbox=(0.0, 0.0, 10.0, 10.0)) -> None:
        self.bbox = bbox
        self.calls = 0

    def detect_bgr(self, image_bgr) -> DetectedFace:
        self.calls += 1
        return DetectedFace(
            embedding=np.ones(512, dtype=np.float32),
            bbox=np.asarray(self.bbox, dtype=np.float32),
            kps=None,
        )


class RecordingWriter:
    def __init__(self) -> None:
        self.upserts: list[tuple[str, np.ndarray]] = []

    def upsert(self, face_id: str, vector: np.ndarray) -> None:
        self.upserts.append((face_id, vector))

    def delete(self, face_id: str) -> bool:
        return True


class FixedIdentity:
    def new_id(self) -> str:
        return "fixed-id"


def _enroll(*, scorer, gate=STRICT_GATE):
    writer = RecordingWriter()
    use_case = EnrollFaceUseCase(
        loader=Base64ImageLoader(OpenCvImageDecoder()),
        detector=StubDetector(),
        gate=FaceGate(metrics=OpenCvFaceImageMetrics(), scorer=scorer),
        writer=writer,
        identity=FixedIdentity(),
    )
    return use_case, writer, gate


# --- Review Focus #1 -------------------------------------------------------
def test_enroll_with_liveness_false_runs_no_checks():
    """liveness=false must skip quality AND anti-spoof, matching today.

    The stub detector reports a 10x10 bbox on a flat grey image, which fails
    min_side, blur and brightness. With liveness=false none of that runs.
    """
    use_case, writer, gate = _enroll(scorer=UnavailableLivenessChecker())
    request = SingleFrameRequest(
        image_base64=_png_base64(), liveness=False, limits=LIMITS, gate=gate
    )
    result = use_case.execute(request)
    assert result.face_id == "fixed-id"
    assert result.liveness_score is None
    assert len(writer.upserts) == 1


# --- Review Focus #5 -------------------------------------------------------
def test_unavailable_liveness_checker_raises_dependency_unavailable():
    """ready() is probed before scoring, so RuntimeError never escapes."""
    use_case, _, gate = _enroll(scorer=UnavailableLivenessChecker())
    relaxed = GateParams(
        blur_min_variance=0.0, brightness_min=0.0, brightness_max=255.0,
        min_side_px=0.0, liveness_threshold=0.5,
    )
    request = SingleFrameRequest(
        image_base64=_png_base64(), liveness=True, limits=LIMITS, gate=relaxed
    )
    with pytest.raises(DependencyUnavailableError) as exc:
        use_case.execute(request)
    assert str(exc.value) == (
        "liveness model not available (run scripts/download_liveness_model.py)"
    )


def test_enroll_with_liveness_true_returns_the_score():
    use_case, writer, _ = _enroll(scorer=FakeLivenessChecker(score=0.91))
    relaxed = GateParams(
        blur_min_variance=0.0, brightness_min=0.0, brightness_max=255.0,
        min_side_px=0.0, liveness_threshold=0.5,
    )
    request = SingleFrameRequest(
        image_base64=_png_base64(), liveness=True, limits=LIMITS, gate=relaxed
    )
    result = use_case.execute(request)
    assert result.liveness_score == pytest.approx(0.91)
    assert writer.upserts[0][0] == "fixed-id"


def test_check_face_takes_no_writer_and_returns_no_score_without_liveness():
    """`/v1/faces/check` validates; it must never be able to persist.

    Enrollment's RecordingWriter test (above) covers the case where a writer
    *is* wired in. Here the guarantee is structural: CheckFaceUseCase cannot
    write because no writer can be handed to it. Asserting on the constructor
    signature pins that, where probing a private attribute would not.
    """
    params = inspect.signature(CheckFaceUseCase.__init__).parameters
    assert not [p for p in params if "writer" in p or "store" in p], (
        f"CheckFaceUseCase must take no writer/store collaborator, got {list(params)}"
    )
    use_case = CheckFaceUseCase(
        loader=Base64ImageLoader(OpenCvImageDecoder()),
        detector=StubDetector(),
        gate=FaceGate(metrics=OpenCvFaceImageMetrics(), scorer=FakeLivenessChecker()),
    )
    request = SingleFrameRequest(
        image_base64=_png_base64(), liveness=False, limits=LIMITS, gate=STRICT_GATE
    )
    assert use_case.execute(request).liveness_score is None


def test_loader_strips_a_data_url_prefix():
    loader = Base64ImageLoader(OpenCvImageDecoder())
    payload = "data:image/png;base64," + _png_base64()
    image = loader.load(payload, limits=LIMITS)
    assert image.shape[2] == 3


def test_loader_rejects_non_base64_with_the_legacy_message():
    loader = Base64ImageLoader(OpenCvImageDecoder())
    with pytest.raises(InvalidImageError) as exc:
        loader.load("!!!not base64!!!", limits=LIMITS)
    assert str(exc.value) == "invalid image_base64"
