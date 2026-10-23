from pydantic import BaseModel, ConfigDict, Field


class FaceImageRequest(BaseModel):
    """JSON body for enroll / update: base64 image + optional thresholds."""

    model_config = ConfigDict(extra="forbid")

    image_base64: str
    liveness: bool = False
    blur_min_variance: float | None = None
    brightness_min: float | None = None
    brightness_max: float | None = None
    min_side_px: float | None = None
    liveness_threshold: float | None = None


class FaceVerifyRequest(FaceImageRequest):
    match_threshold: float | None = None


class FaceSearchRequest(FaceImageRequest):
    limit: int | None = None
    min_score: float | None = None


class FaceListRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    limit: int | None = None
    offset: int = Field(default=0, ge=0)


class GroupsListRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_size: int | None = None


class GroupRebuildRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_score: float | None = None
    min_size: int | None = None
    top_k: int | None = Field(default=None, ge=1)


class HealthResponse(BaseModel):
    status: str
    detail: str | None = None

class FaceIdResponse(BaseModel):
    face_id: str
    liveness_score: float | None = None


class CheckResponse(BaseModel):
    ok: bool = True
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
    run_id: str | None = None
    built_at: str | None = None


class GroupRebuildAccepted(BaseModel):
    run_id: str
    status: str = "running"


class GroupRebuildStatus(BaseModel):
    run_id: str
    status: str
    threshold: float
    top_k: int
    min_size: int
    face_count: int | None = None
    group_count: int | None = None
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None


class GroupRebuildListResponse(BaseModel):
    runs: list[GroupRebuildStatus]
