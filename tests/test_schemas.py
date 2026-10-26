import pytest
from pydantic import ValidationError

from app.api.schemas import FaceLivenessRequest, LivenessResponse


def test_liveness_request_requires_two_frames():
    with pytest.raises(ValidationError):
        FaceLivenessRequest(images_base64=["only-one"])


def test_liveness_request_forbids_unknown_fields():
    with pytest.raises(ValidationError):
        FaceLivenessRequest(images_base64=["a", "b"], liveness=True)


def test_liveness_request_accepts_optional_overrides():
    req = FaceLivenessRequest(
        images_base64=["a", "b"], liveness_min_distance=0.1
    )
    assert req.liveness_min_distance == 0.1
    assert req.liveness_threshold is None


def test_liveness_response_shape():
    resp = LivenessResponse(
        liveness_score=0.9, frame_count=2, inter_frame_distance=0.3
    )
    assert resp.ok is True
    assert resp.model_dump() == {
        "ok": True,
        "liveness_score": 0.9,
        "frame_count": 2,
        "inter_frame_distance": 0.3,
    }
