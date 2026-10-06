from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str
    detail: str | None = None

class FaceIdResponse(BaseModel):
    face_id: str
    liveness_score: float | None = None

class UpdateResponse(BaseModel):
    face_id: str
    updated: bool = True
    liveness_score: float | None = None

class DeleteResponse(BaseModel):
    face_id: str
    deleted: bool = True


class ClearResponse(BaseModel):
    cleared: bool = True
    deleted_count: int


class CountResponse(BaseModel):
    count: int


class FaceExistsResponse(BaseModel):
    face_id: str
    exists: bool = True


class FaceListResponse(BaseModel):
    face_ids: list[str]
    total: int
    limit: int
    offset: int


class StatsResponse(BaseModel):
    faces_count: int
    store_ok: bool
    pipeline_ok: bool


class VerifyResponse(BaseModel):
    face_id: str
    match: bool
    score: float
    threshold: float
    liveness_score: float | None = None

class SearchHit(BaseModel):
    face_id: str
    score: float

class SearchResponse(BaseModel):
    results: list[SearchHit]
    threshold: float
    liveness_score: float | None = None


class FaceGroup(BaseModel):
    face_ids: list[str]
    size: int


class GroupsResponse(BaseModel):
    threshold: float
    groups: list[FaceGroup]
