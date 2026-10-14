import pytest

from app.pipeline.device import resolve_face_device


def test_cpu_always_cpu():
    r = resolve_face_device("cpu", available_providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert r.effective == "cpu"
    assert r.providers == ["CPUExecutionProvider"]
    assert r.ctx_id == -1
    assert r.fallback_reason is None
    assert r.requested == "cpu"


def test_auto_without_cuda_uses_cpu():
    r = resolve_face_device("auto", available_providers=["CPUExecutionProvider"])
    assert r.effective == "cpu"
    assert r.providers == ["CPUExecutionProvider"]
    assert r.ctx_id == -1
    assert r.fallback_reason is not None


def test_auto_with_cuda_uses_cuda():
    r = resolve_face_device(
        "auto", available_providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    assert r.effective == "cuda"
    assert r.providers[0] == "CUDAExecutionProvider"
    assert "CPUExecutionProvider" in r.providers
    assert r.ctx_id == 0
    assert r.fallback_reason is None


def test_cuda_missing_falls_back_to_cpu():
    r = resolve_face_device("cuda", available_providers=["CPUExecutionProvider"])
    assert r.effective == "cpu"
    assert r.ctx_id == -1
    assert r.fallback_reason is not None


def test_invalid_device_raises():
    with pytest.raises(ValueError, match="FACE_DEVICE"):
        resolve_face_device("gpu", available_providers=["CPUExecutionProvider"])
