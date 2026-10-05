import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.errors import (
    DependencyUnavailableError,
    FaceDetectionError,
    FaceNotFoundError,
    InvalidImageError,
)


logger = logging.getLogger(__name__)


def _register_exception_handlers(app: FastAPI) -> None:
    mapping = {
        InvalidImageError: 400,
        FaceNotFoundError: 404,
        FaceDetectionError: 422,
        DependencyUnavailableError: 503,
    }
    for exc_type, status in mapping.items():
        def handler(request: Request, exc: Exception, _status: int = status):
            return JSONResponse(status_code=_status, content={"detail": str(exc)})

        app.add_exception_handler(exc_type, handler)


@asynccontextmanager
async def _production_lifespan(app: FastAPI):
    # Fail fast with a clear message: the service is useless without models and Qdrant.
    from app.config import get_settings
    from app.pipeline.onnx_insightface import InsightFacePipeline
    from app.store.qdrant import QdrantFaceStore

    settings = get_settings()
    try:
        app.state.pipeline = InsightFacePipeline(settings.model_dir)
    except Exception as exc:
        logger.error(
            "Failed to load InsightFace models from %r (run scripts/download_models.py): %s",
            settings.model_dir,
            exc,
        )
        raise RuntimeError(
            f"InsightFace pipeline init failed (model_dir={settings.model_dir!r}): {exc}"
        ) from exc
    try:
        app.state.store = QdrantFaceStore.from_settings(settings)
    except Exception as exc:
        logger.error("Failed to connect to Qdrant at %s: %s", settings.qdrant_url, exc)
        raise RuntimeError(
            f"Qdrant store init failed (url={settings.qdrant_url!r}): {exc}"
        ) from exc
    yield


def create_app(*, testing: bool = False) -> FastAPI:
    app = FastAPI(
        title="iM.FaceId",
        version="0.1.0",
        lifespan=None if testing else _production_lifespan,
    )
    app.state.testing = testing
    if testing:
        from app.pipeline.fake import FakeFacePipeline
        from app.store.memory import MemoryFaceStore

        app.state.pipeline = FakeFacePipeline()
        app.state.store = MemoryFaceStore()
    _register_exception_handlers(app)
    app.include_router(router)
    return app

app = create_app()
