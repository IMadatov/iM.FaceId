# Client Face Parameter Overrides Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let clients override quality, liveness threshold, and verify match threshold per request via optional Form fields (search `min_score` stays Query).

**Architecture:** Resolve `client if not None else settings.*` in route helpers; pass explicit thresholds into `_maybe_face_quality` / `_maybe_liveness` / verify. No clamping. Quality still runs only when `liveness=true`.

**Tech Stack:** FastAPI Form params, existing `Settings`, pytest + TestClient, FakeFacePipeline / FakeLivenessChecker.

**Spec:** `docs/superpowers/specs/2026-10-16-client-face-param-overrides-design.md`

## Global Constraints

- Optional Form floats; omit → server `.env` / Settings default
- Full override (no floors/ceilings); disable semantics unchanged (blur `0`, etc.)
- Search score remains Query `min_score` / `limit`
- Do not expose grouping / ANN / upload-size / crop-scale
- No new response fields

## File structure

| File | Role |
|------|------|
| `app/api/routes.py` | Form params + resolve + wire helpers |
| `tests/test_api_faces.py` | API override tests |
| `README.md` | Document Form overrides |
| `.env.example` | Note that Form can override quality defaults |

---

### Task 1: Resolve helpers + wire routes

**Files:**
- Modify: `app/api/routes.py`
- Test: `tests/test_api_faces.py`

**Interfaces:**
- Produces: `_resolve(client: float | None, default: float) -> float`
- Produces: `_maybe_face_quality(..., blur_min_variance, brightness_min, brightness_max, min_side_px)` — no longer reads quality from `settings` for those knobs
- Produces: `_maybe_liveness(..., threshold: float)` — uses passed threshold
- Form fields: `blur_min_variance`, `brightness_min`, `brightness_max`, `min_side_px`, `liveness_threshold` on enroll/update/search/verify; `match_threshold` on verify only

- [x] **Step 1: Write failing API tests**

Add to `tests/test_api_faces.py`:

```python
def test_enroll_client_blur_override_stricter_422():
    """Server blur disabled; client Form requires blur → quality 422."""
    app = create_app(testing=True)
    app.state.pipeline = FakeFacePipeline()
    app.state.store = MemoryFaceStore()
    _override_settings(
        app,
        face_blur_min_variance=0.0,
        face_brightness_min=0.0,
        face_brightness_max=255.0,
        face_min_side_px=0.0,
    )
    c = TestClient(app)
    r = c.post(
        "/v1/faces",
        files={"image": ("a.png", _blurry_img_bytes(120), "image/png")},
        data={"liveness": "true", "blur_min_variance": "50"},
    )
    assert r.status_code == 422
    detail = r.json()["detail"].lower()
    assert "quality" in detail or "blur" in detail


def test_verify_client_match_threshold():
    app = create_app(testing=True)
    app.state.pipeline = FakeFacePipeline()
    app.state.store = MemoryFaceStore()
    _override_settings(app, face_match_threshold=0.99)
    c = TestClient(app)
    enroll = c.post(
        "/v1/faces",
        files={"image": ("a.png", _img_bytes(120), "image/png")},
    )
    face_id = enroll.json()["face_id"]
    # Same-ish image may score high with FakeFacePipeline; force match with low threshold
    r = c.post(
        f"/v1/faces/{face_id}/verify",
        files={"image": ("b.png", _img_bytes(120), "image/png")},
        data={"match_threshold": "0.0"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["threshold"] == 0.0
    assert body["match"] is True


def test_enroll_client_liveness_threshold_stricter_422():
    app = create_app(testing=True)
    app.state.pipeline = FakeFacePipeline()
    app.state.store = MemoryFaceStore()
    _override_settings(
        app,
        face_blur_min_variance=0.0,
        face_brightness_min=0.0,
        face_brightness_max=255.0,
        face_min_side_px=0.0,
        liveness_threshold=0.50,
    )
    from app.pipeline.fake_liveness import FakeLivenessChecker

    app.state.liveness = FakeLivenessChecker(score=0.60)
    c = TestClient(app)
    r = c.post(
        "/v1/faces",
        files={"image": ("a.png", _img_bytes(120), "image/png")},
        data={"liveness": "true", "liveness_threshold": "0.90"},
    )
    assert r.status_code == 422
    assert "liveness" in r.json()["detail"].lower()
```

- [x] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_api_faces.py::test_enroll_client_blur_override_stricter_422 tests/test_api_faces.py::test_verify_client_match_threshold tests/test_api_faces.py::test_enroll_client_liveness_threshold_stricter_422 -v`

Expected: FAIL (unknown Form fields ignored or thresholds not applied — blur test may pass enroll 201; liveness may pass; match threshold still 0.99)

- [x] **Step 3: Implement helpers and Form wiring**

In `app/api/routes.py`:

```python
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


def _maybe_liveness(
    img, *, liveness: bool, checker, threshold: float, bbox=None
):
    if not liveness:
        return None
    if not checker.ready():
        raise DependencyUnavailableError(
            "liveness model not available "
            "(run scripts/download_liveness_model.py)"
        )
    return checker.ensure_live(img, threshold=threshold, bbox=bbox)
```

On enroll / update / search / verify, add optional Form params and resolve:

```python
blur_min_variance: float | None = Form(None),
brightness_min: float | None = Form(None),
brightness_max: float | None = Form(None),
min_side_px: float | None = Form(None),
liveness_threshold: float | None = Form(None),
# verify only:
match_threshold: float | None = Form(None),
```

Call sites:

```python
_maybe_face_quality(
    img,
    liveness=liveness,
    bbox=detected.bbox,
    blur_min_variance=_resolve(blur_min_variance, settings.face_blur_min_variance),
    brightness_min=_resolve(brightness_min, settings.face_brightness_min),
    brightness_max=_resolve(brightness_max, settings.face_brightness_max),
    min_side_px=_resolve(min_side_px, settings.face_min_side_px),
)
live_score = _maybe_liveness(
    img,
    liveness=liveness,
    checker=checker,
    threshold=_resolve(liveness_threshold, settings.liveness_threshold),
    bbox=detected.bbox,
)
# verify:
threshold = _resolve(match_threshold, settings.face_match_threshold)
match=is_match(score, threshold)
# return threshold=threshold
```

- [x] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_api_faces.py -v`

Expected: PASS (including new override tests and existing liveness/quality tests)

- [x] **Step 5: Commit**

```bash
git add app/api/routes.py tests/test_api_faces.py
git commit -m "feat: allow per-request Form overrides for face thresholds"
```

---

### Task 2: Docs

**Files:**
- Modify: `README.md`
- Modify: `.env.example`

- [x] **Step 1: Document Form overrides in README**

After the liveness paragraph (~line 97), add:

```markdown
Optional Form overrides (omit → env defaults): `blur_min_variance`, `brightness_min`, `brightness_max`, `min_side_px`, `liveness_threshold` on enroll/update/search/verify; `match_threshold` on verify. Search score still uses Query `min_score`. Full override — clients may relax or disable checks (same disable values as env).
```

Update the env table note for quality rows to say “default; overridable per request via Form”.

- [x] **Step 2: Note in `.env.example`**

```bash
# Face quality defaults (liveness=true). Clients may override via Form on each request.
```

- [x] **Step 3: Commit**

```bash
git add README.md .env.example
git commit -m "docs: document per-request face parameter Form overrides"
```

---

## Spec coverage check

| Spec requirement | Task |
|------------------|------|
| Quality Form overrides | Task 1 |
| `liveness_threshold` Form | Task 1 |
| `match_threshold` on verify | Task 1 |
| Search `min_score` stays Query | unchanged / Task 1 docs |
| Full override / omit → default | Task 1 |
| README / `.env.example` | Task 2 |
