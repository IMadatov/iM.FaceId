from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str
    detail: str | None = None

class FaceIdResponse(BaseModel):
    face_id: str

class UpdateResponse(BaseModel):
    face_id: str
    updated: bool = True

class DeleteResponse(BaseModel):
    face_id: str
    deleted: bool = True

class VerifyResponse(BaseModel):
    face_id: str
    match: bool
    score: float
    threshold: float

class SearchHit(BaseModel):
    face_id: str
    score: float

class SearchResponse(BaseModel):
    results: list[SearchHit]
    threshold: float
