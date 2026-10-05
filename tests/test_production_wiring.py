import pytest
from fastapi.testclient import TestClient

from app.main import create_app


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
    monkeypatch.setattr(
        "app.pipeline.onnx_insightface.InsightFacePipeline",
        lambda model_dir, **kwargs: sentinel_pipe,
    )
    monkeypatch.setattr(
        "app.store.qdrant.QdrantFaceStore.from_settings",
        classmethod(lambda cls, settings: sentinel_store),
    )
    app = create_app(testing=False)
    with TestClient(app):
        assert app.state.pipeline is sentinel_pipe
        assert app.state.store is sentinel_store
