# Face Check Endpoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `POST /v1/faces/check` — same image gates as enroll, no store write, no embedding in the response.

**Architecture:** Reuse `FaceImageRequest` and `_decode_base64_image` / `detect_bgr` / `_run_quality_and_liveness`. Return `CheckResponse(ok=True, liveness_score=...)`. Do not inject or call `store`.

**Tech Stack:** FastAPI, Pydantic, pytest + TestClient, FakeFacePipeline / MemoryFaceStore / FakeLivenessChecker.

**Spec:** `docs/superpowers/specs/2026-10-23-face-check-endpoint-design.md`

## Global Constraints

- Body: `FaceImageRequest` (same as enroll)
- Path: `POST /v1/faces/check`
- No `store` side effects; no embedding / `face_id` in response
- Errors identical to enroll for the shared pipeline prefix
- Register `/v1/faces/check` before any `/v1/faces/{face_id}` route
- Do not change enroll / update / search / verify behavior

## File structure

| File | Role |
|------|------|
| `app/api/schemas.py` | `CheckResponse` |
| `app/api/routes.py` | `check_face` handler |
| `tests/test_api_faces.py` | API tests (pass, no store write, quality/liveness) |
| `docs/CLIENT_API.md` | Client docs for the new endpoint |

---

### Task 1: Schema + route + API tests

**Files:**
- Modify: `app/api/schemas.py`
- Modify: `app/api/routes.py`
- Modify: `tests/test_api_faces.py`

**Interfaces:**
- Produces: `CheckResponse(ok: bool = True, liveness_score: float | None = None)`
- Produces: `check_face(body: FaceImageRequest, ...) -> CheckResponse` at `POST /v1/faces/check`
- Consumes: `_decode_base64_image`, `pipeline.detect_bgr`, `_run_quality_and_liveness` (existing)

- [ ] **Step 1: Write failing API tests**

Append to `tests/test_api_faces.py`:

```python
def test_check_face_ok_does_not_store():
    c = _client()
    assert c.get("/v1/faces/count").json()["count"] == 0
    r = c.post("/v1/faces/check", json=_face_json(_img_bytes(120)))
    assert r.status_code == 200
    body = r.json()
    assert body == {"ok": True, "liveness_score": None}
    assert "face_id" not in body
    assert "embedding" not in body
    assert c.get("/v1/faces/count").json()["count"] == 0


def test_check_face_liveness_pass_returns_score():
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
        "/v1/faces/check",
        json=_face_json(_img_bytes(120), liveness=True),
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["liveness_score"] is not None
    assert body["liveness_score"] >= 0.5
    assert c.get("/v1/faces/count").json()["count"] == 0


def test_check_face_blurry_with_liveness_quality_422():
    app = create_app(testing=True)
    app.state.pipeline = FakeFacePipeline()
    app.state.store = MemoryFaceStore()
    _override_settings(
        app,
        face_blur_min_variance=50.0,
        face_brightness_min=0.0,
        face_brightness_max=255.0,
        face_min_side_px=0.0,
    )
    c = TestClient(app)
    r = c.post(
        "/v1/faces/check",
        json=_face_json(_blurry_img_bytes(120), liveness=True),
    )
    assert r.status_code == 422
    detail = r.json()["detail"].lower()
    assert "quality" in detail or "blur" in detail
    assert "liveness" not in detail
    assert c.get("/v1/faces/count").json()["count"] == 0


def test_check_face_liveness_fail_422():
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
    from app.pipeline.fake_liveness import FakeLivenessChecker

    app.state.liveness = FakeLivenessChecker(force_spoof=True)
    c = TestClient(app)
    r = c.post(
        "/v1/faces/check",
        json=_face_json(_img_bytes(120), liveness=True),
    )
    assert r.status_code == 422
    assert "liveness" in r.json()["detail"].lower()
    assert c.get("/v1/faces/count").json()["count"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_api_faces.py::test_check_face_ok_does_not_store \
  tests/test_api_faces.py::test_check_face_liveness_pass_returns_score \
  tests/test_api_faces.py::test_check_face_blurry_with_liveness_quality_422 \
  tests/test_api_faces.py::test_check_face_liveness_fail_422 -v
```

Expected: FAIL (404 Not Found or route missing / response model missing).

- [ ] **Step 3: Add `CheckResponse` schema**

In `app/api/schemas.py`, after `FaceIdResponse`:

```python
class CheckResponse(BaseModel):
    ok: bool = True
    liveness_score: float | None = None
```

- [ ] **Step 4: Add `check_face` route**

In `app/api/routes.py`:

1. Import `CheckResponse` in the schemas import block.
2. Add the handler **immediately after** `enroll_face` (still before any `/v1/faces/{face_id}` routes):

```python
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
```

Do **not** add `store=Depends(store_dep)` or call `store` in any way.

- [ ] **Step 5: Run tests to verify they pass**

Run:

```bash
pytest tests/test_api_faces.py::test_check_face_ok_does_not_store \
  tests/test_api_faces.py::test_check_face_liveness_pass_returns_score \
  tests/test_api_faces.py::test_check_face_blurry_with_liveness_quality_422 \
  tests/test_api_faces.py::test_check_face_liveness_fail_422 -v
```

Expected: all four PASS.

Also run a quick regression on enroll:

```bash
pytest tests/test_api_faces.py::test_enroll_update_verify_search_delete_flow -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add app/api/schemas.py app/api/routes.py tests/test_api_faces.py
git commit -m "$(cat <<'EOF'
feat: add POST /v1/faces/check preflight endpoint

EOF
)"
```

---

### Task 2: Client API docs

**Files:**
- Modify: `docs/CLIENT_API.md`

**Interfaces:**
- Documents: `POST /v1/faces/check` (request = enroll body; response `ok` + `liveness_score`)

- [ ] **Step 1: Insert docs section**

In `docs/CLIENT_API.md`, insert **after** the `POST /v1/faces` — Enroll section (before `PUT /v1/faces/{face_id}`):

```markdown
### `POST /v1/faces/check` — Preflight check

Same image gates as enroll (decode, detect, optional quality/liveness). Does **not** store a face and does **not** return an embedding.

**Request**

Same body as enroll.

**Response 200**

```json
{
  "ok": true,
  "liveness_score": null
}
```

`liveness_score` is set only when `liveness` was `true`.

**curl**

```bash
IMG=$(base64 -w0 face.png)
curl -s -X POST "$BASE/v1/faces/check" \
  -H 'Content-Type: application/json' \
  -d "{\"image_base64\":\"$IMG\",\"liveness\":true}"
```

Errors match enroll for the shared pipeline (invalid image, no face, quality, liveness).
```

Also add a one-line entry under the Endpoints list / TOC if the doc has a bullet index at the top that lists routes; if there is no index, skip.

- [ ] **Step 2: Commit**

```bash
git add docs/CLIENT_API.md
git commit -m "$(cat <<'EOF'
docs: document POST /v1/faces/check

EOF
)"
```

---

## Spec coverage checklist

| Spec requirement | Task |
|------------------|------|
| `POST /v1/faces/check` + `FaceImageRequest` | Task 1 |
| Same decode / detect / quality / liveness as enroll | Task 1 |
| No store write | Task 1 (`test_check_face_ok_does_not_store` + no `store` dep) |
| No embedding / `face_id` in response | Task 1 |
| `CheckResponse` `{ ok, liveness_score }` | Task 1 |
| Errors same as enroll | Task 1 (quality + liveness 422 tests) |
| Route before `{face_id}` | Task 1 (place after enroll) |
| `CLIENT_API.md` | Task 2 |
| Non-goals (no dry_run on enroll, no detect-only pipeline) | Not implemented (intentional) |
