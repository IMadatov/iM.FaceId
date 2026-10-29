from app.application.gate import FaceGate
from app.application.image_loader import Base64ImageLoader
from app.application.params import SingleFrameRequest
from app.application.results import EnrollResult
from app.domain.ports.identity import IdentityFactory
from app.domain.ports.pipeline import FaceDetector
from app.domain.ports.store import FaceWriter


class EnrollFaceUseCase:
    """Decode -> detect -> gate -> mint id -> store."""

    def __init__(
        self,
        *,
        loader: Base64ImageLoader,
        detector: FaceDetector,
        gate: FaceGate,
        writer: FaceWriter,
        identity: IdentityFactory,
    ) -> None:
        self._loader = loader
        self._detector = detector
        self._gate = gate
        self._writer = writer
        self._identity = identity

    def execute(self, request: SingleFrameRequest) -> EnrollResult:
        image = self._loader.load(request.image_base64, limits=request.limits)
        detected = self._detector.detect_bgr(image)
        score = self._gate.run(
            image, detected, liveness=request.liveness, gate=request.gate
        )
        face_id = self._identity.new_id()
        self._writer.upsert(face_id, detected.embedding)
        return EnrollResult(face_id=face_id, liveness_score=score)
