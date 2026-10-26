# Multi-frame Passive Liveness Endpoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new `POST /v1/faces/liveness` endpoint that accepts 2+ frames and passes passive liveness only when the mean per-frame anti-spoof score clears the threshold **and** the frames show enough inter-frame embedding variation (rejecting static photo/screen spoofs).

**Architecture:** A standalone FastAPI route decodes each frame, runs the existing detect → quality → passive-score path per frame, then combines two signals: the mean liveness score (vs `liveness_threshold`) and the maximum pairwise cosine distance between frame embeddings (vs `liveness_min_distance`). Reuses `pipeline.detect_bgr`, `ensure_face_quality`, the ONNX liveness checker, and `cosine_similarity`. The existing `/v1/faces/check` endpoint is untouched; no store writes.

**Tech Stack:** Python, FastAPI, Pydantic v2, NumPy, pytest, OpenCV (test fixtures).

**Spec:** [docs/superpowers/specs/2026-10-26-multiframe-liveness-endpoint-design.md](../specs/2026-10-26-multiframe-liveness-endpoint-design.md)

## Global Constraints

- New endpoint path: `POST /v1/faces/liveness`. Do **not** modify `/v1/faces/check`, enroll, update, verify, or search.
- Minimum frames: `2` (static `min_length=2` on the schema). Maximum frames: `settings.liveness_max_frames` (runtime config, enforced in the handler).
- Always-on liveness: no `liveness` flag on the request; quality gate + score + distance always run.
- No store side effects: the handler does **not** inject or call `store`; no embedding/`face_id` in the response.
- Reuse existing error types and their status mapping: `InvalidImageError→400`, `FaceDetectionError→422`, `ImageQualityError→422`, `LivenessFailedError→422`, `DependencyUnavailableError→503`. No new error types.
- Decision rule (exact): `mean(scores) >= liveness_threshold` AND `max_pairwise(1 - cosine_similarity) >= liveness_min_distance`.
- Config defaults: `liveness_max_frames = 5`, `liveness_min_distance = 0.05` (the latter is **NEEDS TUNING** — do not treat as final).

## Review Focus

- **Frame count exactly at `liveness_max_frames`** must be accepted (only strictly greater is rejected) — pinned in Task 4 boundary test.
- **Unknown request fields** must be rejected with 422 (`extra="forbid"`) — pinned in Task 4 forbid test.
- **A `data:` URI-prefixed base64 frame** must decode like `/check`/enroll do (prefix stripped) — pinned in Task 4 data-URI test.
- **A bad/no-face frame among valid ones** must fail and name which frame index — pinned in Task 4 bad-frame test.
- **Liveness model not installed** must return 503, not 500 — pinned in Task 4 unavailable test.

---

## File Structure

- `app/config.py` — add two `Settings` fields (Task 1).
- `app/api/schemas.py` — add `FaceLivenessRequest`, `LivenessResponse` (Task 2).
- `app/matching/cosine.py` — add `cosine_distance` helper (Task 3).
- `app/api/routes.py` — add `check_liveness` handler + `_run_multiframe_liveness` helper; extend `app.errors` import (Task 4).
- `tests/test_api_faces.py` — endpoint tests (Task 4); `tests/test_cosine.py` or existing matching test — `cosine_distance` test (Task 3).
- `docs/CLIENT_API.md`, `README.md` — document the endpoint (Task 5).

---

### Task 1: Settings — multi-frame liveness config

**Files:**
- Modify: `app/config.py` (add fields near existing `liveness_*` at lines 22-24)
- Test: `tests/test_config.py` (create if absent)

**Interfaces:**
- Produces: `Settings.liveness_max_frames: int` (default `5`), `Settings.liveness_min_distance: float` (default `0.05`).

- [ ] **Step 1: Write the failing test**

Create or append to `tests/test_config.py`:

```python
from app.config import Settings


def test_multiframe_liveness_defaults():
    s = Settings()
    assert s.liveness_max_frames == 5
    assert s.liveness_min_distance == 0.05
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py::test_multiframe_liveness_defaults -v`
Expected: FAIL — `AttributeError: 'Settings' object has no attribute 'liveness_max_frames'`

- [ ] **Step 3: Add the settings**

In `app/config.py`, after line 24 (`liveness_crop_scale: float = 2.7`):

```python
    liveness_max_frames: int = 5
    liveness_min_distance: float = 0.05  # NEEDS TUNING: min max-pairwise cosine distance
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py::test_multiframe_liveness_defaults -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/config.py tests/test_config.py
git commit -m "feat: add multi-frame liveness settings"
```

---

### Task 2: Schemas — request and response models

**Files:**
- Modify: `app/api/schemas.py`
- Test: `tests/test_schemas.py` (create if absent)

**Interfaces:**
- Produces:
  - `FaceLivenessRequest(images_base64: list[str], blur_min_variance: float|None, brightness_min: float|None, brightness_max: float|None, min_side_px: float|None, liveness_threshold: float|None, liveness_min_distance: float|None)` — `extra="forbid"`, `images_base64` has `min_length=2`.
  - `LivenessResponse(ok: bool = True, liveness_score: float, frame_count: int, inter_frame_distance: float)`.

- [ ] **Step 1: Write the failing test**

Create or append to `tests/test_schemas.py`:

```python
import pytest
from pydantic import ValidationError

from app.api.schemas import FaceLivenessRequest, LivenessResponse


def test_liveness_request_requires_two_frames():
    with pytest.raises(ValidationError):
        FaceLivenessRequest(images_base64=["only-one"])


def test_liveness_request_forbids_unknown_fields():
    with pytest.raises(ValidationError):
        FaceLivenessRequest(images_base64=["a", "b"], liveness=True)


def test_liveness_request_accepts_optional_overrides():
    req = FaceLivenessRequest(
        images_base64=["a", "b"], liveness_min_distance=0.1
    )
    assert req.liveness_min_distance == 0.1
    assert req.liveness_threshold is None


def test_liveness_response_shape():
    resp = LivenessResponse(
        liveness_score=0.9, frame_count=2, inter_frame_distance=0.3
    )
    assert resp.ok is True
    assert resp.model_dump() == {
        "ok": True,
        "liveness_score": 0.9,
        "frame_count": 2,
        "inter_frame_distance": 0.3,
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_schemas.py -v`
Expected: FAIL — `ImportError: cannot import name 'FaceLivenessRequest'`

- [ ] **Step 3: Add the models**

In `app/api/schemas.py`, after `CheckResponse` (line 59):

```python
class FaceLivenessRequest(BaseModel):
    """JSON body for multi-frame passive liveness check."""

    model_config = ConfigDict(extra="forbid")

    images_base64: list[str] = Field(..., min_length=2)
    blur_min_variance: float | None = None
    brightness_min: float | None = None
    brightness_max: float | None = None
    min_side_px: float | None = None
    liveness_threshold: float | None = None
    liveness_min_distance: float | None = None


class LivenessResponse(BaseModel):
    ok: bool = True
    liveness_score: float
    frame_count: int
    inter_frame_distance: float
```

(`BaseModel`, `ConfigDict`, `Field` are already imported at the top of the file.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_schemas.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add app/api/schemas.py tests/test_schemas.py
git commit -m "feat: add FaceLivenessRequest and LivenessResponse schemas"
```

---

### Task 3: Cosine distance helper

**Files:**
- Modify: `app/matching/cosine.py`
- Test: `tests/test_cosine.py` (create if absent)

**Interfaces:**
- Consumes: `cosine_similarity(a: np.ndarray, b: np.ndarray) -> float` (existing).
- Produces: `cosine_distance(a: np.ndarray, b: np.ndarray) -> float` returning `1.0 - cosine_similarity(a, b)`.

- [ ] **Step 1: Write the failing test**

Create or append to `tests/test_cosine.py`:

```python
import numpy as np

from app.matching.cosine import cosine_distance


def test_cosine_distance_identical_is_zero():
    v = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    assert cosine_distance(v, v) == 0.0


def test_cosine_distance_orthogonal_is_one():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert abs(cosine_distance(a, b) - 1.0) < 1e-6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cosine.py -v`
Expected: FAIL — `ImportError: cannot import name 'cosine_distance'`

- [ ] **Step 3: Add the helper**

In `app/matching/cosine.py`, after `cosine_similarity` (line 11):

```python
def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - cosine_similarity(a, b)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cosine.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add app/matching/cosine.py tests/test_cosine.py
git commit -m "feat: add cosine_distance helper"
```

---

### Task 4: `/v1/faces/liveness` endpoint

**Files:**
- Modify: `app/api/routes.py` (imports at lines 6-46; add handler after `check_face`, line 168)
- Test: `tests/test_api_faces.py` (append at end; reuse existing `_img_bytes`, `_override_settings`, `create_app`, `FakeFacePipeline`, `MemoryFaceStore`, `TestClient`)

**Interfaces:**
- Consumes: `FaceLivenessRequest`, `LivenessResponse` (Task 2); `Settings.liveness_max_frames`, `Settings.liveness_min_distance` (Task 1); `cosine_distance` (Task 3); existing `_decode_base64_image`, `_resolve`, `ensure_face_quality`, `pipeline_dep`, `settings_dep`, `liveness_dep`.
- Produces: `POST /v1/faces/liveness` returning `LivenessResponse`; module helper `_run_multiframe_liveness(images_base64: list[str], body: FaceLivenessRequest, settings, pipeline, checker) -> tuple[float, float]` returning `(mean_score, max_distance)`.

- [ ] **Step 1: Write the failing happy-path test**

Append to `tests/test_api_faces.py`:

```python
def _liveness_json(images: list[bytes], **kwargs):
    body = {
        "images_base64": [base64.b64encode(i).decode("ascii") for i in images]
    }
    body.update(kwargs)
    return body


def _liveness_app(**settings_kwargs):
    app = create_app(testing=True)
    app.state.pipeline = FakeFacePipeline()
    app.state.store = MemoryFaceStore()
    _override_settings(
        app,
        face_blur_min_variance=0.0,
        face_brightness_min=0.0,
        face_brightness_max=255.0,
        face_min_side_px=0.0,
        **settings_kwargs,
    )
    return app


def test_liveness_distinct_frames_pass():
    app = _liveness_app()
    c = TestClient(app)
    r = c.post(
        "/v1/faces/liveness",
        json=_liveness_json([_img_bytes(120), _img_bytes(121), _img_bytes(122)]),
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["frame_count"] == 3
    assert body["liveness_score"] >= 0.5
    assert body["inter_frame_distance"] > 0.05
    assert c.get("/v1/faces/count").json()["count"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_api_faces.py::test_liveness_distinct_frames_pass -v`
Expected: FAIL — 404 (route not registered) / assertion error

- [ ] **Step 3: Extend imports in `app/api/routes.py`**

Change the schema import block (lines 6-31) to also import the new models — add these names to the existing `from app.api.schemas import (...)` list:

```python
    FaceLivenessRequest,
    LivenessResponse,
```

Change the errors import (line 42) to:

```python
from app.errors import (
    DependencyUnavailableError,
    FaceDetectionError,
    FaceNotFoundError,
    InvalidImageError,
    LivenessFailedError,
)
```

Change the cosine import (line 45) to:

```python
from app.matching.cosine import cosine_distance, cosine_similarity, is_match
```

- [ ] **Step 4: Add the helper and handler**

In `app/api/routes.py`, after `check_face` (ends line 168), add:

```python
def _run_multiframe_liveness(images_base64, body, settings, pipeline, checker):
    if not checker.ready():
        raise DependencyUnavailableError(
            "liveness model not available "
            "(run scripts/download_liveness_model.py)"
        )
    if len(images_base64) > settings.liveness_max_frames:
        raise HTTPException(
            status_code=422,
            detail=f"too many frames (max {settings.liveness_max_frames})",
        )
    blur_min_variance = _resolve(
        body.blur_min_variance, settings.face_blur_min_variance
    )
    brightness_min = _resolve(body.brightness_min, settings.face_brightness_min)
    brightness_max = _resolve(body.brightness_max, settings.face_brightness_max)
    min_side_px = _resolve(body.min_side_px, settings.face_min_side_px)
    threshold = _resolve(body.liveness_threshold, settings.liveness_threshold)
    min_distance = _resolve(
        body.liveness_min_distance, settings.liveness_min_distance
    )

    scores: list[float] = []
    embeddings = []
    for index, image_base64 in enumerate(images_base64):
        try:
            img = _decode_base64_image(image_base64, settings)
        except InvalidImageError as exc:
            raise InvalidImageError(f"frame[{index}]: {exc}") from exc
        try:
            detected = pipeline.detect_bgr(img)
        except FaceDetectionError as exc:
            raise FaceDetectionError(
                f"frame[{index}]: {exc}", code=exc.code
            ) from exc
        ensure_face_quality(
            img,
            detected.bbox,
            blur_min_variance=blur_min_variance,
            brightness_min=brightness_min,
            brightness_max=brightness_max,
            min_side_px=min_side_px,
        )
        scores.append(checker.score_bgr(img, bbox=detected.bbox))
        embeddings.append(detected.embedding)

    mean_score = float(sum(scores) / len(scores))
    max_distance = max(
        cosine_distance(embeddings[i], embeddings[j])
        for i in range(len(embeddings))
        for j in range(i + 1, len(embeddings))
    )
    if mean_score < threshold:
        raise LivenessFailedError(
            f"liveness check failed (mean score={mean_score:.4f} "
            f"< {threshold:.4f})"
        )
    if max_distance < min_distance:
        raise LivenessFailedError(
            f"liveness check failed (frames too similar: "
            f"distance={max_distance:.4f} < {min_distance:.4f})"
        )
    return mean_score, max_distance


@router.post("/v1/faces/liveness", response_model=LivenessResponse)
def check_liveness(
    body: FaceLivenessRequest,
    settings: Settings = Depends(settings_dep),
    pipeline=Depends(pipeline_dep),
    checker=Depends(liveness_dep),
) -> LivenessResponse:
    """Passive multi-frame liveness: mean score + inter-frame variation."""
    mean_score, max_distance = _run_multiframe_liveness(
        body.images_base64, body, settings, pipeline, checker
    )
    return LivenessResponse(
        ok=True,
        liveness_score=mean_score,
        frame_count=len(body.images_base64),
        inter_frame_distance=max_distance,
    )
```

- [ ] **Step 5: Run the happy-path test to verify it passes**

Run: `pytest tests/test_api_faces.py::test_liveness_distinct_frames_pass -v`
Expected: PASS

- [ ] **Step 6: Add the remaining endpoint tests**

Append to `tests/test_api_faces.py`:

```python
def test_liveness_identical_frames_rejected_as_too_similar():
    app = _liveness_app()
    c = TestClient(app)
    frame = _img_bytes(120)
    r = c.post("/v1/faces/liveness", json=_liveness_json([frame, frame]))
    assert r.status_code == 422
    assert "similar" in r.json()["detail"].lower()
    assert c.get("/v1/faces/count").json()["count"] == 0


def test_liveness_low_mean_score_rejected():
    from app.pipeline.fake_liveness import FakeLivenessChecker

    app = _liveness_app()
    app.state.liveness = FakeLivenessChecker(force_spoof=True)
    c = TestClient(app)
    r = c.post(
        "/v1/faces/liveness",
        json=_liveness_json([_img_bytes(120), _img_bytes(121)]),
    )
    assert r.status_code == 422
    assert "score" in r.json()["detail"].lower()


def test_liveness_single_frame_rejected_by_validation():
    app = _liveness_app()
    c = TestClient(app)
    r = c.post("/v1/faces/liveness", json=_liveness_json([_img_bytes(120)]))
    assert r.status_code == 422


def test_liveness_too_many_frames_rejected():
    app = _liveness_app()  # liveness_max_frames default = 5
    c = TestClient(app)
    frames = [_img_bytes(120 + i) for i in range(6)]
    r = c.post("/v1/faces/liveness", json=_liveness_json(frames))
    assert r.status_code == 422
    assert "many" in r.json()["detail"].lower()


def test_liveness_exactly_max_frames_pass():
    app = _liveness_app()  # max = 5
    c = TestClient(app)
    frames = [_img_bytes(120 + i) for i in range(5)]
    r = c.post("/v1/faces/liveness", json=_liveness_json(frames))
    assert r.status_code == 200
    assert r.json()["frame_count"] == 5


def test_liveness_unknown_field_rejected():
    app = _liveness_app()
    c = TestClient(app)
    body = _liveness_json([_img_bytes(120), _img_bytes(121)])
    body["liveness"] = True
    r = c.post("/v1/faces/liveness", json=body)
    assert r.status_code == 422


def test_liveness_bad_frame_names_index():
    app = _liveness_app()
    c = TestClient(app)
    # second frame is all-black -> FakeFacePipeline raises "no face"
    r = c.post(
        "/v1/faces/liveness",
        json=_liveness_json([_img_bytes(120), _img_bytes(0)]),
    )
    assert r.status_code == 422
    assert "frame[1]" in r.json()["detail"]


def test_liveness_data_uri_prefix_decodes():
    app = _liveness_app()
    c = TestClient(app)
    b64 = base64.b64encode(_img_bytes(120)).decode("ascii")
    body = {
        "images_base64": [
            f"data:image/png;base64,{b64}",
            base64.b64encode(_img_bytes(121)).decode("ascii"),
        ]
    }
    r = c.post("/v1/faces/liveness", json=body)
    assert r.status_code == 200
    assert r.json()["frame_count"] == 2


def test_liveness_model_unavailable_503():
    from app.pipeline.fake_liveness import UnavailableLivenessChecker

    app = _liveness_app()
    app.state.liveness = UnavailableLivenessChecker()
    c = TestClient(app)
    r = c.post(
        "/v1/faces/liveness",
        json=_liveness_json([_img_bytes(120), _img_bytes(121)]),
    )
    assert r.status_code == 503
```

- [ ] **Step 7: Run all endpoint tests to verify they pass**

Run: `pytest tests/test_api_faces.py -k liveness -v`
Expected: PASS (all `*liveness*` tests, including the new ones)

- [ ] **Step 8: Run the full suite (no regressions)**

Run: `pytest -q`
Expected: PASS (existing `/check`, enroll, verify, search tests unchanged)

- [ ] **Step 9: Commit**

```bash
git add app/api/routes.py tests/test_api_faces.py
git commit -m "feat: add POST /v1/faces/liveness multi-frame liveness endpoint"
```

---

### Task 5: Documentation

**Files:**
- Modify: `docs/CLIENT_API.md`, `README.md`

**Interfaces:**
- Consumes: the final request/response contract from Task 2 and Task 4.

- [ ] **Step 1: Document the endpoint in `docs/CLIENT_API.md`**

Add a section next to the `/v1/faces/check` docs:

```markdown
### POST /v1/faces/liveness

Passive multi-frame liveness. Send 2–`liveness_max_frames` (default 5) frames
of the same person. Passes only when the mean per-frame anti-spoof score
clears `liveness_threshold` **and** the frames vary enough between each other
(max pairwise embedding cosine distance ≥ `liveness_min_distance`), which
rejects a static photo or screen held to the camera. No face is stored.

**Request**

```json
{
  "images_base64": ["<base64>", "<base64>"],
  "blur_min_variance": null,
  "brightness_min": null,
  "brightness_max": null,
  "min_side_px": null,
  "liveness_threshold": null,
  "liveness_min_distance": null
}
```

Optional fields override the server defaults when non-null.

**Response 200**

```json
{ "ok": true, "liveness_score": 0.97, "frame_count": 3, "inter_frame_distance": 0.11 }
```

**Errors:** `400` invalid/oversized frame; `422` validation (fewer than 2 or
more than `liveness_max_frames` frames, unknown field), no/multiple faces in a
frame, quality failure, mean score below threshold, or frames too similar;
`503` liveness model not installed.
```

- [ ] **Step 2: Add a line to `README.md`**

In the endpoint list, add `POST /v1/faces/liveness` with a one-line description ("multi-frame passive liveness; rejects static photo/screen") next to `/v1/faces/check`.

- [ ] **Step 3: Commit**

```bash
git add docs/CLIENT_API.md README.md
git commit -m "docs: document POST /v1/faces/liveness"
```

- [ ] **Step 4: Update the graph**

Run: `graphify update .`

```bash
git add graphify-out
git commit -m "chore: update knowledge graph for liveness endpoint"
```

---

## Self-Review Notes

- **Spec coverage:** §2 goals → Tasks 2/4 (request shape, mean score, distance, always-on, no store, response, docs/tests); §5 contract → Tasks 2/4; §6 config → Task 1; §7 files → Tasks 1–5; §8 tests → Task 4. All sections mapped.
- **Type consistency:** `_run_multiframe_liveness(...) -> (mean_score, max_distance)` matches handler unpacking; `cosine_distance` signature matches Task 3; `FaceLivenessRequest`/`LivenessResponse` field names identical across Tasks 2 and 4.
- **Review Focus:** boundary (`test_liveness_exactly_max_frames_pass`), forbid (`test_liveness_unknown_field_rejected`), data URI (`test_liveness_data_uri_prefix_decodes`), bad frame index (`test_liveness_bad_frame_names_index`), model unavailable (`test_liveness_model_unavailable_503`) — all pinned in Task 4.
