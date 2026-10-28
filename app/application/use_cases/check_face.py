from app.application.gate import FaceGate
from app.application.image_loader import Base64ImageLoader
from app.application.params import SingleFrameRequest
from app.application.results import CheckResult
from app.domain.ports.pipeline import FaceDetector


class CheckFaceUseCase:
    """Preflight: the same gates as enroll, with no store write."""

    def __init__(
        self,
        *,
        loader: Base64ImageLoader,
        detector: FaceDetector,
        gate: FaceGate,
    ) -> None:
        self._loader = loader
        self._detector = detector
        self._gate = gate

    def execute(self, request: SingleFrameRequest) -> CheckResult:
        image = self._loader.load(request.image_base64, limits=request.limits)
        detected = self._detector.detect_bgr(image)
        score = self._gate.run(
            image, detected, liveness=request.liveness, gate=request.gate
        )
        return CheckResult(liveness_score=score)
