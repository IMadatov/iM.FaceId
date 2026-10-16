from fastapi import Request

from app.config import Settings, get_settings
from app.errors import DependencyUnavailableError
from app.jobs.group_rebuild import GroupRebuildService


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


def session_factory_dep(request: Request):
    factory = getattr(request.app.state, "session_factory", None)
    if factory is None:
        raise DependencyUnavailableError("database not configured")
    return factory


def group_rebuild_dep(request: Request) -> GroupRebuildService:
    service = getattr(request.app.state, "group_rebuild", None)
    if service is None:
        raise DependencyUnavailableError("group rebuild service not configured")
    return service
