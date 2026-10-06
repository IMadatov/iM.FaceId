from unittest.mock import MagicMock

import numpy as np
import pytest

from app.errors import LivenessFailedError
from app.pipeline.onnx_liveness import OnnxLivenessChecker, _softmax, crop_face_bgr


def test_softmax_sums_to_one():
    p = _softmax(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert p.shape == (3,)
    assert float(p.sum()) == pytest.approx(1.0, abs=1e-5)


def test_crop_face_bgr_expands_around_bbox_and_stays_in_bounds():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    img[40:60, 80:120] = 255  # face region
    bbox = np.array([80.0, 40.0, 120.0, 60.0], dtype=np.float32)
    crop = crop_face_bgr(img, bbox, scale=2.7)
    # Expanded ~2.7x around 40x20 box → larger than raw bbox, within image
    assert crop.ndim == 3
    assert crop.shape[0] > 20
    assert crop.shape[1] > 40
    assert crop.shape[0] <= 100
    assert crop.shape[1] <= 200
    # Bright face pixels must still be present after expansion
    assert int(crop.max()) == 255
    assert float(crop.mean()) > float(img.mean())


def test_crop_face_bgr_scale_one_is_tight_bbox():
    img = np.arange(100 * 100 * 3, dtype=np.uint8).reshape(100, 100, 3)
    bbox = np.array([10.0, 20.0, 50.0, 60.0], dtype=np.float32)
    crop = crop_face_bgr(img, bbox, scale=1.0)
    # Silent-Face uses inclusive bottom-right (+1), so size is ~40–41
    assert 40 <= crop.shape[0] <= 41
    assert 40 <= crop.shape[1] <= 41
    assert float(crop.mean()) == pytest.approx(float(img[20:60, 10:50].mean()), abs=5.0)


def _mock_checker(monkeypatch, tmp_path, logits):
    model = tmp_path / "dummy.onnx"
    model.write_bytes(b"x")
    session = MagicMock()
    inp = MagicMock()
    inp.name = "input"
    session.get_inputs.return_value = [inp]
    session.run.return_value = [np.asarray(logits, dtype=np.float32)]
    monkeypatch.setattr("onnxruntime.InferenceSession", lambda *a, **k: session)
    return OnnxLivenessChecker(str(model), crop_scale=2.7), session


def test_onnx_liveness_ensure_live_with_mock_session(monkeypatch, tmp_path):
    # Silent-Face: class 1 = live
    checker, session = _mock_checker(
        monkeypatch, tmp_path, [[0.1, 5.0, 0.1]]
    )
    img = np.full((120, 120, 3), 180, dtype=np.uint8)
    score = checker.ensure_live(img, threshold=0.5)
    assert score > 0.5
    session.run.assert_called_once()
    fed = session.run.call_args[0][1]["input"]
    assert fed.shape == (1, 3, 80, 80)
    # Raw BGR float 0–255 (no /255)
    assert float(fed.mean()) == pytest.approx(180.0, abs=1.0)


def test_onnx_liveness_with_bbox_feeds_80x80_tensor(monkeypatch, tmp_path):
    checker, session = _mock_checker(
        monkeypatch, tmp_path, [[0.1, 5.0, 0.1]]
    )
    img = np.full((200, 200, 3), 10, dtype=np.uint8)
    img[70:130, 70:130] = 200
    bbox = np.array([70.0, 70.0, 130.0, 130.0], dtype=np.float32)
    score = checker.ensure_live(img, threshold=0.5, bbox=bbox)
    assert score > 0.5
    fed = session.run.call_args[0][1]["input"]
    assert fed.shape == (1, 3, 80, 80)
    full = checker._preprocess(img)
    assert float(fed.mean()) > float(full.mean())


def test_onnx_liveness_fails_when_spoof_logits(monkeypatch, tmp_path):
    # High mass on attack classes 0 and 2
    checker, _ = _mock_checker(monkeypatch, tmp_path, [[5.0, 0.1, 5.0]])
    img = np.full((120, 120, 3), 180, dtype=np.uint8)
    with pytest.raises(LivenessFailedError):
        checker.ensure_live(img, threshold=0.5)
