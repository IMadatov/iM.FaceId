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
