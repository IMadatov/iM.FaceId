import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile

from app.api.schemas import (
    ClearResponse,
    CountResponse,
    DeleteResponse,
    FaceExistsResponse,
    FaceGroup,
    FaceIdResponse,
    FaceListResponse,
    GroupsResponse,
    HealthResponse,
    SearchHit,
    SearchResponse,
    StatsResponse,
    UpdateResponse,
    VerifyResponse,
)
from app.config import Settings
from app.deps import liveness_dep, pipeline_dep, settings_dep, store_dep
from app.errors import DependencyUnavailableError, FaceNotFoundError
from app.imaging import decode_and_resize
from app.matching.cosine import cosine_similarity, is_match
from app.matching.grouping import group_similar_faces

router = APIRouter()


async def _decode_upload(image: UploadFile, settings: Settings):
    data = await image.read()
    return decode_and_resize(
        data,
        max_side=settings.max_image_side,
        max_bytes=settings.max_upload_bytes,
    )


def _maybe_liveness(img, *, liveness: bool, settings: Settings, checker, bbox=None):
    if not liveness:
        return None
    if not checker.ready():
        raise DependencyUnavailableError(
            "liveness model not available "
            "(run scripts/download_liveness_model.py)"
        )
    return checker.ensure_live(
        img, threshold=settings.liveness_threshold, bbox=bbox
    )


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
    liveness: bool = Form(False),
    settings: Settings = Depends(settings_dep),
    pipeline=Depends(pipeline_dep),
    store=Depends(store_dep),
    checker=Depends(liveness_dep),
) -> FaceIdResponse:
    img = await _decode_upload(image, settings)
    detected = pipeline.detect_bgr(img)
    live_score = _maybe_liveness(
        img,
        liveness=liveness,
        settings=settings,
        checker=checker,
        bbox=detected.bbox,
    )
    face_id = str(uuid.uuid4())
    store.upsert(face_id, detected.embedding)
    return FaceIdResponse(face_id=face_id, liveness_score=live_score)


@router.delete("/v1/faces", response_model=ClearResponse)
def clear_faces(store=Depends(store_dep)) -> ClearResponse:
    """Remove all enrolled face embeddings (manual full wipe)."""
    deleted_count = store.clear()
    return ClearResponse(cleared=True, deleted_count=deleted_count)


@router.get("/v1/faces/count", response_model=CountResponse)
def count_faces(store=Depends(store_dep)) -> CountResponse:
    return CountResponse(count=store.count())


@router.get("/v1/faces", response_model=FaceListResponse)
def list_faces(
    limit: int | None = Query(default=None),
    offset: int = Query(default=0, ge=0),
    settings: Settings = Depends(settings_dep),
    store=Depends(store_dep),
) -> FaceListResponse:
    effective = settings.face_list_default_limit if limit is None else limit
    effective = max(1, min(effective, settings.face_list_max_limit))
    return FaceListResponse(
        face_ids=store.list_ids(limit=effective, offset=offset),
        total=store.count(),
        limit=effective,
        offset=offset,
    )


@router.get("/v1/stats", response_model=StatsResponse)
def stats(
    pipeline=Depends(pipeline_dep), store=Depends(store_dep)
) -> StatsResponse:
    return StatsResponse(
        faces_count=store.count(),
        store_ok=store.ping(),
        pipeline_ok=pipeline.ready(),
    )


@router.get("/v1/faces/groups", response_model=GroupsResponse)
def list_face_groups(
    min_score: float | None = Query(default=None),
    min_size: int | None = Query(default=None),
    settings: Settings = Depends(settings_dep),
    store=Depends(store_dep),
) -> GroupsResponse:
    threshold = settings.face_search_min_score if min_score is None else min_score
    size = settings.face_groups_min_size if min_size is None else min_size
    if size < 2:
        raise HTTPException(status_code=400, detail="min_size must be >= 2")
    items = store.list_all()
    if len(items) > settings.face_groups_max_faces:
        raise HTTPException(
            status_code=400,
            detail=(
                f"too many faces for on-demand grouping "
                f"({len(items)} > {settings.face_groups_max_faces})"
            ),
        )
    groups = group_similar_faces(items, min_score=threshold, min_size=size)
    return GroupsResponse(
        threshold=threshold,
        groups=[FaceGroup(face_ids=g, size=len(g)) for g in groups],
    )


@router.post("/v1/faces/search", response_model=SearchResponse)
async def search_faces(
    image: UploadFile = File(...),
    liveness: bool = Form(False),
    limit: int | None = Query(default=None),
    min_score: float | None = Query(default=None),
    settings: Settings = Depends(settings_dep),
    pipeline=Depends(pipeline_dep),
    store=Depends(store_dep),
    checker=Depends(liveness_dep),
) -> SearchResponse:
    effective_limit = settings.face_search_default_limit if limit is None else limit
    effective_limit = max(1, min(effective_limit, settings.face_search_max_limit))
    threshold = settings.face_search_min_score if min_score is None else min_score
    img = await _decode_upload(image, settings)
    detected = pipeline.detect_bgr(img)
    live_score = _maybe_liveness(
        img,
        liveness=liveness,
        settings=settings,
        checker=checker,
        bbox=detected.bbox,
    )
    hits = store.search(
        detected.embedding, limit=effective_limit, min_score=threshold
    )
    return SearchResponse(
        results=[SearchHit(face_id=fid, score=score) for fid, score in hits],
        threshold=threshold,
        liveness_score=live_score,
    )


@router.put("/v1/faces/{face_id}", response_model=UpdateResponse)
async def update_face(
    face_id: str,
    image: UploadFile = File(...),
    liveness: bool = Form(False),
    settings: Settings = Depends(settings_dep),
    pipeline=Depends(pipeline_dep),
    store=Depends(store_dep),
    checker=Depends(liveness_dep),
) -> UpdateResponse:
    if store.get(face_id) is None:
        raise FaceNotFoundError(f"face {face_id} not found")
    img = await _decode_upload(image, settings)
    detected = pipeline.detect_bgr(img)
    live_score = _maybe_liveness(
        img,
        liveness=liveness,
        settings=settings,
        checker=checker,
        bbox=detected.bbox,
    )
    store.upsert(face_id, detected.embedding)
    return UpdateResponse(face_id=face_id, updated=True, liveness_score=live_score)


@router.get("/v1/faces/{face_id}", response_model=FaceExistsResponse)
def get_face(face_id: str, store=Depends(store_dep)) -> FaceExistsResponse:
    if store.get(face_id) is None:
        raise FaceNotFoundError(f"face {face_id} not found")
    return FaceExistsResponse(face_id=face_id, exists=True)


@router.delete("/v1/faces/{face_id}", response_model=DeleteResponse)
def delete_face(face_id: str, store=Depends(store_dep)) -> DeleteResponse:
    if not store.delete(face_id):
        raise FaceNotFoundError(f"face {face_id} not found")
    return DeleteResponse(face_id=face_id, deleted=True)


@router.post("/v1/faces/{face_id}/verify", response_model=VerifyResponse)
async def verify_face(
    face_id: str,
    image: UploadFile = File(...),
    liveness: bool = Form(False),
    settings: Settings = Depends(settings_dep),
    pipeline=Depends(pipeline_dep),
    store=Depends(store_dep),
    checker=Depends(liveness_dep),
) -> VerifyResponse:
    stored = store.get(face_id)
    if stored is None:
        raise FaceNotFoundError(f"face {face_id} not found")
    img = await _decode_upload(image, settings)
    detected = pipeline.detect_bgr(img)
    live_score = _maybe_liveness(
        img,
        liveness=liveness,
        settings=settings,
        checker=checker,
        bbox=detected.bbox,
    )
    score = cosine_similarity(stored, detected.embedding)
    return VerifyResponse(
        face_id=face_id,
        match=is_match(score, settings.face_match_threshold),
        score=score,
        threshold=settings.face_match_threshold,
        liveness_score=live_score,
    )
