from types import SimpleNamespace

import numpy as np
import pytest

from app.errors import FaceDetectionError
from app.pipeline.onnx_insightface import InsightFacePipeline


class _StubApp:
    def __init__(self, faces):
        self._faces = faces

    def get(self, image_bgr):
        return self._faces


def _pipeline(faces) -> InsightFacePipeline:
    pipe = InsightFacePipeline.__new__(InsightFacePipeline)
    pipe._app = _StubApp(faces)
    pipe._ready = True
    pipe._dim = 512
    return pipe


IMG = np.zeros((4, 4, 3), dtype=np.uint8)
BBOX = np.array([0.0, 0.0, 3.0, 3.0], dtype=np.float32)


def _face(embedding, bbox=BBOX):
    return SimpleNamespace(normed_embedding=embedding, bbox=bbox)


def test_embed_happy_path_shape_and_dtype():
    emb = np.random.default_rng(0).standard_normal(512)
    out = _pipeline([_face(emb)]).embed_bgr(IMG)
    assert out.shape == (512,)
    assert out.dtype == np.float32


def test_detect_bgr_returns_embedding_and_bbox():
    emb = np.ones(512, dtype=np.float32)
    bbox = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    detected = _pipeline([_face(emb, bbox)]).detect_bgr(IMG)
    assert detected.embedding.shape == (512,)
    np.testing.assert_array_equal(detected.bbox, bbox)


def test_detect_bgr_missing_bbox_raises():
    with pytest.raises(FaceDetectionError, match="bbox"):
        _pipeline([SimpleNamespace(normed_embedding=np.ones(512))]).detect_bgr(IMG)


def test_embed_none_embedding_raises():
    with pytest.raises(FaceDetectionError):
        _pipeline([_face(None)]).embed_bgr(IMG)


def test_embed_missing_embedding_attr_raises():
    with pytest.raises(FaceDetectionError):
        _pipeline([SimpleNamespace(bbox=BBOX)]).embed_bgr(IMG)


@pytest.mark.parametrize(
    "bad",
    [np.zeros(0), np.ones(128), np.ones((2, 512))],
    ids=["empty", "wrong_dim", "2d"],
)
def test_embed_wrong_shape_raises(bad):
    with pytest.raises(FaceDetectionError, match="shape"):
        _pipeline([_face(bad)]).embed_bgr(IMG)


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_embed_non_finite_raises(value):
    emb = np.ones(512)
    emb[3] = value
    with pytest.raises(FaceDetectionError, match="non-finite"):
        _pipeline([_face(emb)]).embed_bgr(IMG)


def test_embed_respects_configured_dim():
    pipe = _pipeline([_face(np.ones(128))])
    pipe._dim = 128
    assert pipe.embed_bgr(IMG).shape == (128,)


def test_embed_no_face_raises():
    with pytest.raises(FaceDetectionError, match="no face"):
        _pipeline([]).embed_bgr(IMG)


def test_embed_multiple_faces_raises():
    faces = [_face(np.ones(512))] * 2
    with pytest.raises(FaceDetectionError, match="multiple"):
        _pipeline(faces).embed_bgr(IMG)
