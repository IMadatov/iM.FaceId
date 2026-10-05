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
    return pipe


IMG = np.zeros((4, 4, 3), dtype=np.uint8)


def test_embed_happy_path_shape_and_dtype():
    emb = np.random.default_rng(0).standard_normal(512)
    out = _pipeline([SimpleNamespace(normed_embedding=emb)]).embed_bgr(IMG)
    assert out.shape == (512,)
    assert out.dtype == np.float32


def test_embed_none_embedding_raises():
    with pytest.raises(FaceDetectionError):
        _pipeline([SimpleNamespace(normed_embedding=None)]).embed_bgr(IMG)


def test_embed_missing_embedding_attr_raises():
    with pytest.raises(FaceDetectionError):
        _pipeline([SimpleNamespace()]).embed_bgr(IMG)


def test_embed_no_face_raises():
    with pytest.raises(FaceDetectionError, match="no face"):
        _pipeline([]).embed_bgr(IMG)


def test_embed_multiple_faces_raises():
    faces = [SimpleNamespace(normed_embedding=np.ones(512))] * 2
    with pytest.raises(FaceDetectionError, match="multiple"):
        _pipeline(faces).embed_bgr(IMG)
