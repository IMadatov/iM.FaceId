from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.errors import (
    DependencyUnavailableError,
    FaceDetectionError,
    FaceNotFoundError,
    InvalidImageError,
)


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


def create_app(*, testing: bool = False) -> FastAPI:
    app = FastAPI(title="iM.FaceId", version="0.1.0")
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
