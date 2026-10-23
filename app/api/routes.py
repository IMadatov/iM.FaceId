import base64
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from app.api.schemas import (
    CheckResponse,
    ClearResponse,
    CountResponse,
    DeleteResponse,
    FaceExistsResponse,
    FaceGroup,
    FaceIdResponse,
    FaceImageRequest,
    FaceListRequest,
    FaceListResponse,
    FaceSearchRequest,
    FaceVerifyRequest,
    GroupRebuildAccepted,
    GroupRebuildListResponse,
    GroupRebuildRequest,
    GroupRebuildStatus,
    GroupsListRequest,
    GroupsResponse,
    HealthResponse,
    SearchHit,
    SearchResponse,
    StatsResponse,
    UpdateResponse,
    VerifyResponse,
)
from app.config import Settings
from app.db.repository import GroupsRepository
from app.deps import (
    group_rebuild_dep,
    liveness_dep,
    pipeline_dep,
    session_factory_dep,
    settings_dep,
    store_dep,
)
from app.errors import DependencyUnavailableError, FaceNotFoundError, InvalidImageError
from app.imaging import decode_and_resize
from app.jobs.group_rebuild import GroupRebuildConflict, GroupRebuildService
from app.matching.cosine import cosine_similarity, is_match
from app.pipeline.image_quality import ensure_face_quality

router = APIRouter()


def _iso(dt) -> str | None:
    return None if dt is None else dt.isoformat()


def _decode_base64_image(image_base64: str, settings: Settings):
    raw = image_base64.strip()
    if raw.startswith("data:") and "," in raw:
        raw = raw.split(",", 1)[1]
    try:
        data = base64.b64decode(raw, validate=False)
    except Exception as exc:
        raise InvalidImageError("invalid image_base64") from exc
    return decode_and_resize(
        data,
        max_side=settings.max_image_side,
        max_bytes=settings.max_upload_bytes,
    )


def _resolve(client: float | None, default: float) -> float:
    return default if client is None else client


def _maybe_face_quality(
    img,
    *,
    liveness: bool,
    bbox,
    blur_min_variance: float,
    brightness_min: float,
    brightness_max: float,
    min_side_px: float,
):
    if not liveness:
        return
    ensure_face_quality(
        img,
        bbox,
        blur_min_variance=blur_min_variance,
        brightness_min=brightness_min,
        brightness_max=brightness_max,
        min_side_px=min_side_px,
    )


def _maybe_liveness(img, *, liveness: bool, checker, threshold: float, bbox=None):
    if not liveness:
        return None
    if not checker.ready():
        raise DependencyUnavailableError(
            "liveness model not available "
            "(run scripts/download_liveness_model.py)"
        )
    return checker.ensure_live(img, threshold=threshold, bbox=bbox)


def _run_quality_and_liveness(img, detected, body: FaceImageRequest, settings, checker):
    _maybe_face_quality(
        img,
        liveness=body.liveness,
        bbox=detected.bbox,
        blur_min_variance=_resolve(
            body.blur_min_variance, settings.face_blur_min_variance
        ),
        brightness_min=_resolve(body.brightness_min, settings.face_brightness_min),
        brightness_max=_resolve(body.brightness_max, settings.face_brightness_max),
        min_side_px=_resolve(body.min_side_px, settings.face_min_side_px),
    )
    return _maybe_liveness(
        img,
        liveness=body.liveness,
        checker=checker,
        threshold=_resolve(body.liveness_threshold, settings.liveness_threshold),
        bbox=detected.bbox,
    )


@router.get("/health", response_model=HealthResponse)
def health(
    pipeline=Depends(pipeline_dep), store=Depends(store_dep)
) -> HealthResponse:
    if not pipeline.ready() or not store.ping():
        raise HTTPException(status_code=503, detail="dependencies not ready")
    return HealthResponse(status="ok")


@router.post("/v1/faces", response_model=FaceIdResponse, status_code=201)
def enroll_face(
    body: FaceImageRequest,
    settings: Settings = Depends(settings_dep),
    pipeline=Depends(pipeline_dep),
    store=Depends(store_dep),
    checker=Depends(liveness_dep),
) -> FaceIdResponse:
    img = _decode_base64_image(body.image_base64, settings)
    detected = pipeline.detect_bgr(img)
    live_score = _run_quality_and_liveness(
        img, detected, body, settings, checker
    )
    face_id = str(uuid.uuid4())
    store.upsert(face_id, detected.embedding)
    return FaceIdResponse(face_id=face_id, liveness_score=live_score)


@router.post("/v1/faces/check", response_model=CheckResponse)
def check_face(
    body: FaceImageRequest,
    settings: Settings = Depends(settings_dep),
    pipeline=Depends(pipeline_dep),
    checker=Depends(liveness_dep),
) -> CheckResponse:
    """Preflight: same gates as enroll, no store write."""
    img = _decode_base64_image(body.image_base64, settings)
    detected = pipeline.detect_bgr(img)
    live_score = _run_quality_and_liveness(
        img, detected, body, settings, checker
    )
    return CheckResponse(ok=True, liveness_score=live_score)


@router.delete("/v1/faces", response_model=ClearResponse)
def clear_faces(store=Depends(store_dep)) -> ClearResponse:
    """Remove all enrolled face embeddings (manual full wipe)."""
    deleted_count = store.clear()
    return ClearResponse(cleared=True, deleted_count=deleted_count)


@router.get("/v1/faces/count", response_model=CountResponse)
def count_faces(store=Depends(store_dep)) -> CountResponse:
    return CountResponse(count=store.count())


@router.post("/v1/faces/list", response_model=FaceListResponse)
def list_faces(
    body: FaceListRequest,
    settings: Settings = Depends(settings_dep),
    store=Depends(store_dep),
) -> FaceListResponse:
    effective = settings.face_list_default_limit if body.limit is None else body.limit
    effective = max(1, min(effective, settings.face_list_max_limit))
    return FaceListResponse(
        face_ids=store.list_ids(limit=effective, offset=body.offset),
        total=store.count(),
        limit=effective,
        offset=body.offset,
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


@router.post(
    "/v1/faces/groups/rebuild",
    response_model=GroupRebuildAccepted,
    status_code=202,
)
def rebuild_face_groups(
    body: GroupRebuildRequest,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(settings_dep),
    store=Depends(store_dep),
    rebuild: GroupRebuildService = Depends(group_rebuild_dep),
) -> GroupRebuildAccepted:
    try:
        run_id = rebuild.start(
            store=store,
            settings=settings,
            min_score=body.min_score,
            min_size=body.min_size,
            top_k=body.top_k,
        )
    except GroupRebuildConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    background_tasks.add_task(rebuild.execute, run_id, store=store, settings=settings)
    return GroupRebuildAccepted(run_id=run_id, status="running")


def _group_rebuild_status(run) -> GroupRebuildStatus:
    return GroupRebuildStatus(
        run_id=run.id,
        status=run.status.value,
        threshold=run.threshold,
        top_k=run.top_k,
        min_size=run.min_size,
        face_count=run.face_count,
        group_count=run.group_count,
        error=run.error,
        started_at=_iso(run.started_at),
        finished_at=_iso(run.finished_at),
    )


@router.get(
    "/v1/faces/groups/rebuild",
    response_model=GroupRebuildListResponse,
)
def list_group_rebuild_runs(
    session_factory=Depends(session_factory_dep),
) -> GroupRebuildListResponse:
    session = session_factory()
    try:
        runs = GroupsRepository(session).list_runs()
        return GroupRebuildListResponse(runs=[_group_rebuild_status(r) for r in runs])
    finally:
        session.close()


@router.get(
    "/v1/faces/groups/rebuild/{run_id}",
    response_model=GroupRebuildStatus,
)
def get_group_rebuild_status(
    run_id: str,
    session_factory=Depends(session_factory_dep),
) -> GroupRebuildStatus:
    session = session_factory()
    try:
        run = GroupsRepository(session).get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"group run {run_id} not found")
        return _group_rebuild_status(run)
    finally:
        session.close()


@router.post("/v1/faces/groups/list", response_model=GroupsResponse)
def list_face_groups(
    body: GroupsListRequest,
    settings: Settings = Depends(settings_dep),
    session_factory=Depends(session_factory_dep),
) -> GroupsResponse:
    size = settings.face_groups_min_size if body.min_size is None else body.min_size
    if size < 2:
        raise HTTPException(status_code=400, detail="min_size must be >= 2")
    session = session_factory()
    try:
        snapshot = GroupsRepository(session).latest_completed()
        if snapshot is None:
            return GroupsResponse(
                threshold=settings.face_search_min_score,
                groups=[],
                run_id=None,
                built_at=None,
            )
        groups = [
            FaceGroup(face_ids=g.face_ids, size=g.size)
            for g in snapshot.groups
            if g.size >= size
        ]
        return GroupsResponse(
            threshold=snapshot.threshold,
            groups=groups,
            run_id=snapshot.run_id,
            built_at=_iso(snapshot.built_at),
        )
    finally:
        session.close()


@router.post("/v1/faces/search", response_model=SearchResponse)
def search_faces(
    body: FaceSearchRequest,
    settings: Settings = Depends(settings_dep),
    pipeline=Depends(pipeline_dep),
    store=Depends(store_dep),
    checker=Depends(liveness_dep),
) -> SearchResponse:
    effective_limit = (
        settings.face_search_default_limit if body.limit is None else body.limit
    )
    effective_limit = max(1, min(effective_limit, settings.face_search_max_limit))
    threshold = (
        settings.face_search_min_score if body.min_score is None else body.min_score
    )
    img = _decode_base64_image(body.image_base64, settings)
    detected = pipeline.detect_bgr(img)
    live_score = _run_quality_and_liveness(
        img, detected, body, settings, checker
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
def update_face(
    face_id: str,
    body: FaceImageRequest,
    settings: Settings = Depends(settings_dep),
    pipeline=Depends(pipeline_dep),
    store=Depends(store_dep),
    checker=Depends(liveness_dep),
) -> UpdateResponse:
    if store.get(face_id) is None:
        raise FaceNotFoundError(f"face {face_id} not found")
    img = _decode_base64_image(body.image_base64, settings)
    detected = pipeline.detect_bgr(img)
    live_score = _run_quality_and_liveness(
        img, detected, body, settings, checker
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
def verify_face(
    face_id: str,
    body: FaceVerifyRequest,
    settings: Settings = Depends(settings_dep),
    pipeline=Depends(pipeline_dep),
    store=Depends(store_dep),
    checker=Depends(liveness_dep),
) -> VerifyResponse:
    stored = store.get(face_id)
    if stored is None:
        raise FaceNotFoundError(f"face {face_id} not found")
    img = _decode_base64_image(body.image_base64, settings)
    detected = pipeline.detect_bgr(img)
    live_score = _run_quality_and_liveness(
        img, detected, body, settings, checker
    )
    score = cosine_similarity(stored, detected.embedding)
    threshold = _resolve(body.match_threshold, settings.face_match_threshold)
    return VerifyResponse(
        face_id=face_id,
        match=is_match(score, threshold),
        score=score,
        threshold=threshold,
        liveness_score=live_score,
    )
