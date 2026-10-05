import uuid

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from app.api.schemas import (
    DeleteResponse,
    FaceIdResponse,
    HealthResponse,
    SearchHit,
    SearchResponse,
    UpdateResponse,
    VerifyResponse,
)
from app.config import Settings
from app.deps import pipeline_dep, settings_dep, store_dep
from app.errors import FaceNotFoundError
from app.imaging import decode_and_resize
from app.matching.cosine import cosine_similarity, is_match

router = APIRouter()


async def _embed_upload(image: UploadFile, settings: Settings, pipeline):
    data = await image.read()
    img = decode_and_resize(
        data,
        max_side=settings.max_image_side,
        max_bytes=settings.max_upload_bytes,
    )
    return pipeline.embed_bgr(img)


@router.get("/health", response_model=HealthResponse)
def health(
    pipeline=Depends(pipeline_dep), store=Depends(store_dep)
) -> HealthResponse:
    if not pipeline.ready() or not store.ping():
        raise HTTPException(status_code=503, detail="dependencies not ready")
    return HealthResponse(status="ok")


@router.post("/v1/faces", response_model=FaceIdResponse, status_code=201)
async def enroll_face(
    image: UploadFile = File(...),
    settings: Settings = Depends(settings_dep),
    pipeline=Depends(pipeline_dep),
    store=Depends(store_dep),
) -> FaceIdResponse:
    vector = await _embed_upload(image, settings, pipeline)
    face_id = str(uuid.uuid4())
    store.upsert(face_id, vector)
    return FaceIdResponse(face_id=face_id)


@router.post("/v1/faces/search", response_model=SearchResponse)
async def search_faces(
    image: UploadFile = File(...),
    limit: int | None = Query(default=None),
    min_score: float | None = Query(default=None),
    settings: Settings = Depends(settings_dep),
    pipeline=Depends(pipeline_dep),
    store=Depends(store_dep),
) -> SearchResponse:
    effective_limit = settings.face_search_default_limit if limit is None else limit
    effective_limit = max(1, min(effective_limit, settings.face_search_max_limit))
    threshold = settings.face_search_min_score if min_score is None else min_score
    vector = await _embed_upload(image, settings, pipeline)
    hits = store.search(vector, limit=effective_limit, min_score=threshold)
    return SearchResponse(
        results=[SearchHit(face_id=fid, score=score) for fid, score in hits],
        threshold=threshold,
    )


@router.put("/v1/faces/{face_id}", response_model=UpdateResponse)
async def update_face(
    face_id: str,
    image: UploadFile = File(...),
    settings: Settings = Depends(settings_dep),
    pipeline=Depends(pipeline_dep),
    store=Depends(store_dep),
) -> UpdateResponse:
    if store.get(face_id) is None:
        raise FaceNotFoundError(f"face {face_id} not found")
    vector = await _embed_upload(image, settings, pipeline)
    store.upsert(face_id, vector)
    return UpdateResponse(face_id=face_id, updated=True)


@router.delete("/v1/faces/{face_id}", response_model=DeleteResponse)
def delete_face(face_id: str, store=Depends(store_dep)) -> DeleteResponse:
    if not store.delete(face_id):
        raise FaceNotFoundError(f"face {face_id} not found")
    return DeleteResponse(face_id=face_id, deleted=True)


@router.post("/v1/faces/{face_id}/verify", response_model=VerifyResponse)
async def verify_face(
    face_id: str,
    image: UploadFile = File(...),
    settings: Settings = Depends(settings_dep),
    pipeline=Depends(pipeline_dep),
    store=Depends(store_dep),
) -> VerifyResponse:
    stored = store.get(face_id)
    if stored is None:
        raise FaceNotFoundError(f"face {face_id} not found")
    probe = await _embed_upload(image, settings, pipeline)
    score = cosine_similarity(stored, probe)
    return VerifyResponse(
        face_id=face_id,
        match=is_match(score, settings.face_match_threshold),
        score=score,
        threshold=settings.face_match_threshold,
    )
