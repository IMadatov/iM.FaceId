from fastapi import FastAPI
from app.api.routes import router

def create_app(*, testing: bool = False) -> FastAPI:
    app = FastAPI(title="iM.FaceId", version="0.1.0")
    app.state.testing = testing
    app.include_router(router)
    return app

app = create_app()
