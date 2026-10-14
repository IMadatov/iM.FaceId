from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.errors import (
    DependencyUnavailableError,
    FaceDetectionError,
    FaceNotFoundError,
    InvalidImageError,
    LivenessFailedError,
)

if TYPE_CHECKING:
    from app.pipeline.device import ResolvedDevice
    from app.pipeline.onnx_insightface import InsightFacePipeline


logger = logging.getLogger(__name__)


def build_insightface_with_fallback(
    settings, resolved: ResolvedDevice
) -> tuple[InsightFacePipeline, ResolvedDevice]:
    from app.pipeline.device import ResolvedDevice
    from app.pipeline.onnx_insightface import InsightFacePipeline

    try:
        pipe = InsightFacePipeline(
            settings.model_dir,
            embedding_dim=settings.embedding_dim,
            providers=resolved.providers,
            ctx_id=resolved.ctx_id,
        )
        return pipe, resolved
    except Exception as exc:
        if resolved.effective != "cuda":
            raise
        logger.warning(
            "CUDA InsightFace init failed (%s); falling back to CPU", exc
        )
        cpu = ResolvedDevice(
            requested=resolved.requested,
            effective="cpu",
            providers=["CPUExecutionProvider"],
            ctx_id=-1,
            fallback_reason=f"InsightFace CUDA init failed: {exc}",
        )
        pipe = InsightFacePipeline(
            settings.model_dir,
            embedding_dim=settings.embedding_dim,
            providers=cpu.providers,
            ctx_id=cpu.ctx_id,
        )
        return pipe, cpu


def _register_exception_handlers(app: FastAPI) -> None:
    mapping = {
        InvalidImageError: 400,
        FaceNotFoundError: 404,
        FaceDetectionError: 422,
        LivenessFailedError: 422,
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
    from app.pipeline.device import resolve_face_device
    from app.store.qdrant import QdrantFaceStore

    settings = get_settings()
    try:
        resolved = resolve_face_device(settings.face_device)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    logger.info(
        "face device requested=%s effective=%s fallback=%s",
        resolved.requested,
        resolved.effective,
        resolved.fallback_reason,
    )
    if resolved.fallback_reason:
        logger.warning(
            "FACE_DEVICE fallback requested=%s effective=%s reason=%s",
            resolved.requested,
            resolved.effective,
            resolved.fallback_reason,
        )
    try:
        app.state.pipeline, resolved = build_insightface_with_fallback(
            settings, resolved
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
    from app.pipeline.onnx_liveness import load_liveness_checker

    app.state.liveness = load_liveness_checker(
        settings.liveness_model_path,
        crop_scale=settings.liveness_crop_scale,
        providers=resolved.providers,
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
    _register_exception_handlers(app)
    app.include_router(router)
    return app

app = create_app()
