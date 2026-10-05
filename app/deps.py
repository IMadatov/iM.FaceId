from fastapi import Request

from app.config import Settings, get_settings
from app.errors import DependencyUnavailableError


def settings_dep() -> Settings:
    return get_settings()


def pipeline_dep(request: Request):
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise DependencyUnavailableError("face pipeline not configured")
    return pipeline


def store_dep(request: Request):
    store = getattr(request.app.state, "store", None)
    if store is None:
        raise DependencyUnavailableError("face store not configured")
    return store


def liveness_dep(request: Request):
    checker = getattr(request.app.state, "liveness", None)
    if checker is None:
        raise DependencyUnavailableError("liveness checker not configured")
    return checker
