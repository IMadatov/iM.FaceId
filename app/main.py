import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import router
from sqlalchemy import update

from app.db.models import GroupRun, GroupRunStatus
from app.db.session import create_db_engine, init_db, make_session_factory
from app.errors import (
    DependencyUnavailableError,
    FaceDetectionError,
    FaceNotFoundError,
    ImageQualityError,
    InvalidImageError,
    LivenessFailedError,
)
from app.jobs.group_rebuild import GroupRebuildService


logger = logging.getLogger(__name__)


def _register_exception_handlers(app: FastAPI) -> None:
    mapping = {
        InvalidImageError: 400,
        FaceNotFoundError: 404,
        FaceDetectionError: 422,
        ImageQualityError: 422,
        LivenessFailedError: 422,
        DependencyUnavailableError: 503,
    }
    for exc_type, status in mapping.items():
        def handler(request: Request, exc: Exception, _status: int = status):
            return JSONResponse(status_code=_status, content={"detail": str(exc)})

        app.add_exception_handler(exc_type, handler)


def _wire_database(app: FastAPI, database_url: str) -> None:
    engine = create_db_engine(database_url)
    init_db(engine)
    session_factory = make_session_factory(engine)
    app.state.db_engine = engine
    app.state.session_factory = session_factory
    app.state.group_rebuild = GroupRebuildService(session_factory)

    # Mark interrupted runs so a restart cannot block rebuild forever.
    session = session_factory()
    try:
        session.execute(
            update(GroupRun)
            .where(GroupRun.status.in_((GroupRunStatus.pending, GroupRunStatus.running)))
            .values(status=GroupRunStatus.failed, error="interrupted by process restart")
        )
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@asynccontextmanager
async def _production_lifespan(app: FastAPI):
    # Fail fast with a clear message: the service is useless without models and Qdrant.
    from app.config import get_settings
    from app.pipeline.onnx_insightface import InsightFacePipeline
    from app.store.qdrant import QdrantFaceStore

    settings = get_settings()
    try:
        app.state.pipeline = InsightFacePipeline(
            settings.model_dir, embedding_dim=settings.embedding_dim
        )
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
    try:
        _wire_database(app, settings.database_url)
    except Exception as exc:
        logger.error("Failed to connect to database at %s: %s", settings.database_url, exc)
        raise RuntimeError(
            f"Database init failed (url={settings.database_url!r}): {exc}"
        ) from exc
    from app.pipeline.onnx_liveness import load_liveness_checker

    app.state.liveness = load_liveness_checker(
        settings.liveness_model_path, crop_scale=settings.liveness_crop_scale
    )
    if not app.state.liveness.ready():
        logger.warning(
            "Liveness model not loaded from %r — requests with liveness=true will return 503. "
            "Run: python scripts/download_liveness_model.py",
            settings.liveness_model_path,
        )
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
        from app.pipeline.fake_liveness import FakeLivenessChecker
        from app.store.memory import MemoryFaceStore

        app.state.pipeline = FakeFacePipeline()
        app.state.store = MemoryFaceStore()
        app.state.liveness = FakeLivenessChecker()
        _wire_database(app, "sqlite+pysqlite:///:memory:")
    _register_exception_handlers(app)
    app.include_router(router)
    return app

app = create_app()
