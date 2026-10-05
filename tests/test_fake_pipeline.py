import numpy as np
import pytest
from app.errors import FaceDetectionError
from app.pipeline.fake import FakeFacePipeline


def test_embed_deterministic_and_unit_norm():
    pipe = FakeFacePipeline()
    img = np.full((64, 64, 3), 120, dtype=np.uint8)
    a = pipe.embed_bgr(img)
    b = pipe.embed_bgr(img)
    assert a.shape == (512,)
    assert a.dtype == np.float32
    assert np.allclose(a, b)
    assert np.linalg.norm(a) == pytest.approx(1.0, abs=1e-5)


def test_blank_image_no_face():
    pipe = FakeFacePipeline()
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    with pytest.raises(FaceDetectionError, match="no face detected"):
        pipe.embed_bgr(img)


def test_faces_count_zero_raises_no_face():
    pipe = FakeFacePipeline(faces_count=0)
    img = np.full((64, 64, 3), 120, dtype=np.uint8)
    with pytest.raises(FaceDetectionError, match="no face detected"):
        pipe.embed_bgr(img)


def test_faces_count_multiple_raises():
    pipe = FakeFacePipeline(faces_count=2)
    img = np.full((64, 64, 3), 120, dtype=np.uint8)
    with pytest.raises(FaceDetectionError, match="multiple faces detected"):
        pipe.embed_bgr(img)


def test_ready_returns_true():
    pipe = FakeFacePipeline()
    assert pipe.ready() is True
