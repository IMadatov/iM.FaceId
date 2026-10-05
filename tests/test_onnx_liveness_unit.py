from unittest.mock import MagicMock

import numpy as np
import pytest

from app.errors import LivenessFailedError
from app.pipeline.onnx_liveness import OnnxLivenessChecker, _softmax


def test_softmax_sums_to_one():
    p = _softmax(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert p.shape == (3,)
    assert float(p.sum()) == pytest.approx(1.0, abs=1e-5)


def test_onnx_liveness_ensure_live_with_mock_session(monkeypatch, tmp_path):
    model = tmp_path / "dummy.onnx"
    model.write_bytes(b"not-a-real-model")

    session = MagicMock()
    inp = MagicMock()
    inp.name = "input"
    session.get_inputs.return_value = [inp]
    session.run.return_value = [np.array([[5.0, 0.1, 0.1]], dtype=np.float32)]

    monkeypatch.setattr(
        "onnxruntime.InferenceSession",
        lambda *a, **k: session,
    )
    checker = OnnxLivenessChecker(str(model))
    img = np.full((120, 120, 3), 180, dtype=np.uint8)
    score = checker.ensure_live(img, threshold=0.5)
    assert score > 0.5
    session.run.assert_called_once()


def test_onnx_liveness_fails_when_spoof_logits(monkeypatch, tmp_path):
    model = tmp_path / "dummy.onnx"
    model.write_bytes(b"x")
    session = MagicMock()
    inp = MagicMock()
    inp.name = "input"
    session.get_inputs.return_value = [inp]
    session.run.return_value = [np.array([[0.1, 5.0, 5.0]], dtype=np.float32)]
    monkeypatch.setattr("onnxruntime.InferenceSession", lambda *a, **k: session)
    checker = OnnxLivenessChecker(str(model))
    img = np.full((120, 120, 3), 180, dtype=np.uint8)
    with pytest.raises(LivenessFailedError):
        checker.ensure_live(img, threshold=0.5)
