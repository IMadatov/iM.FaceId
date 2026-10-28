import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.pipeline.device import ResolvedDevice


def test_production_startup_fails_on_invalid_face_device(monkeypatch):
    from app.config import Settings

    monkeypatch.setattr(
        "app.config.get_settings",
        lambda: Settings(face_device="gpu", model_dir="."),
    )
    app = create_app(testing=False)
    with pytest.raises((RuntimeError, ValueError), match="FACE_DEVICE"):
        with TestClient(app):
            pass


def test_production_startup_fails_clearly_when_pipeline_init_fails(monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("models missing")

    monkeypatch.setattr("app.pipeline.onnx_insightface.InsightFacePipeline.__init__", boom)
    app = create_app(testing=False)
    with pytest.raises(RuntimeError, match="InsightFace pipeline init failed"):
        with TestClient(app):
            pass


def test_production_attaches_pipeline_and_store(monkeypatch):
    sentinel_pipe, sentinel_store = object(), object()

    class _Ready:
        def ready(self):
            return True

    sentinel_live = _Ready()
    monkeypatch.setattr(
        "app.pipeline.onnx_insightface.InsightFacePipeline",
        lambda model_dir, **kwargs: sentinel_pipe,
    )
    monkeypatch.setattr(
        "app.store.qdrant.QdrantFaceStore.from_settings",
        classmethod(lambda cls, settings: sentinel_store),
    )
    monkeypatch.setattr(
        "app.pipeline.onnx_liveness.load_liveness_checker",
        lambda *args, **kwargs: sentinel_live,
    )
    def fake_wire(app, database_url):
        app.state.session_factory = object()
        app.state.group_rebuild = object()

    monkeypatch.setattr("app.main._wire_database", fake_wire)
    app = create_app(testing=False)
    with TestClient(app):
        assert app.state.pipeline is sentinel_pipe
        assert app.state.store is sentinel_store
        assert app.state.liveness is sentinel_live
        assert app.state.session_factory is not None
        assert app.state.group_rebuild is not None


def test_production_retries_cuda_init_on_cpu_and_uses_cpu_for_liveness(
    monkeypatch, caplog
):
    from app.config import Settings

    settings = Settings(face_device="cuda", model_dir=".")
    cuda = ResolvedDevice(
        requested="cuda",
        effective="cuda",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        ctx_id=0,
    )
    sentinel_pipe, sentinel_store = object(), object()
    pipeline_calls = []
    liveness_calls = []

    class _Ready:
        def ready(self):
            return True

    def construct_pipeline(model_dir, **kwargs):
        pipeline_calls.append(kwargs)
        if kwargs["providers"][0] == "CUDAExecutionProvider":
            raise OSError("CUDA session failed")
        return sentinel_pipe

    def load_liveness(*args, **kwargs):
        liveness_calls.append(kwargs)
        return _Ready()

    monkeypatch.setattr("app.config.get_settings", lambda: settings)
    monkeypatch.setattr("app.pipeline.device.resolve_face_device", lambda _: cuda)
    monkeypatch.setattr(
        "app.pipeline.onnx_insightface.InsightFacePipeline", construct_pipeline
    )
    monkeypatch.setattr(
        "app.store.qdrant.QdrantFaceStore.from_settings",
        classmethod(lambda cls, settings: sentinel_store),
    )
    monkeypatch.setattr(
        "app.pipeline.onnx_liveness.load_liveness_checker", load_liveness
    )

    with caplog.at_level("WARNING", logger="app.main"):
        app = create_app(testing=False)
        with TestClient(app):
            assert app.state.pipeline is sentinel_pipe

    assert [call["providers"] for call in pipeline_calls] == [
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        ["CPUExecutionProvider"],
    ]
    assert [call["ctx_id"] for call in pipeline_calls] == [0, -1]
    assert liveness_calls[0]["providers"] == ["CPUExecutionProvider"]
    assert "CUDA InsightFace init failed" in caplog.text


def test_production_warns_when_device_resolution_falls_back(monkeypatch, caplog):
    from app.config import Settings

    cpu_fallback = ResolvedDevice(
        requested="auto",
        effective="cpu",
        providers=["CPUExecutionProvider"],
        ctx_id=-1,
        fallback_reason="CUDAExecutionProvider unavailable",
    )

    class _Ready:
        def ready(self):
            return True

    monkeypatch.setattr("app.config.get_settings", lambda: Settings(model_dir="."))
    monkeypatch.setattr(
        "app.pipeline.device.resolve_face_device", lambda _: cpu_fallback
    )
    monkeypatch.setattr(
        "app.pipeline.onnx_insightface.InsightFacePipeline",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "app.store.qdrant.QdrantFaceStore.from_settings",
        classmethod(lambda cls, settings: object()),
    )
    monkeypatch.setattr(
        "app.pipeline.onnx_liveness.load_liveness_checker",
        lambda *args, **kwargs: _Ready(),
    )

    with caplog.at_level("WARNING", logger="app.main"):
        with TestClient(create_app(testing=False)):
            pass

    assert (
        "FACE_DEVICE fallback requested=auto effective=cpu "
        "reason=CUDAExecutionProvider unavailable"
    ) in caplog.text
