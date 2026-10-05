# iM.FaceId — On-Premise CPU FaceID Web API (Design)

**Date:** 2026-10-05  
**Status:** Approved (conversational design)  
**Project:** iM.FaceId

## 1. Intent

Build an internal FaceID microservice that replaces external MyId calls for student attendance verification. MVP is enroll-then-verify only: store a face embedding under a generated ID, then compare a new image against that ID (1:1).

**Out of scope for MVP:** teacher mobile app changes, financial/attendance business logic, liveness/anti-spoofing, request queue workers, MyId parallel comparison, 1:N search.

## 2. Goals and success criteria

| Goal | Success |
|------|---------|
| Independent of MyId for face match | Enroll + verify work on local CPU + Redis |
| Fast enough for classroom use | Target ~60–120 ms per verify on mid-range CPU (best-effort in MVP) |
| Easy manual test | OpenAPI Swagger UI at `/docs` with Try it out |
| Durable embeddings | Vectors survive API process restart (Redis) |

## 3. Architecture

Single FastAPI service (`im-faceid`) with layered modules:

```
Client (Swagger / future backend)
    → API (routes, validation)
    → Pipeline (detect → embed via ONNX)
    → Store (Redis: face_id → vector)
    → Matching (cosine similarity + threshold)
```

**Approach (approved):** Minimal FastAPI — inference in-process (thread/process pool as needed), embeddings in Redis. No Celery/RQ in MVP; can add later without changing the public API shape.

## 4. API

Base path: `/v1`. Interactive docs: `/docs` (Swagger), `/redoc`.

### `POST /v1/faces`

- **Request:** `multipart/form-data` field `image` (file).
- **Behavior:** Validate → resize so max side ≤ `MAX_IMAGE_SIDE` → detect face → compute 512-d embedding → store under new UUID → discard image bytes.
- **Response:** `201 Created`
  ```json
  { "face_id": "<uuid>" }
  ```

### `POST /v1/faces/{face_id}/verify`

- **Request:** path `face_id`, `multipart/form-data` field `image`.
- **Behavior:** Load stored vector → embed probe image → cosine similarity → `match = score >= threshold`.
- **Response:** `200 OK`
  ```json
  {
    "face_id": "<uuid>",
    "match": true,
    "score": 0.72,
    "threshold": 0.40
  }
  ```

### `GET /health`

- **Response:** `200` when models loaded and Redis ping succeeds; otherwise `503` with reason.

### Error responses

| Condition | Status |
|-----------|--------|
| Invalid/oversized file | `400` |
| Unknown `face_id` | `404` |
| No face / multiple faces | `422` |
| Model or Redis unavailable | `503` |

## 5. Model stack (CPU)

| Stage | Model | Runtime |
|-------|--------|---------|
| Detection | SCRFD lightweight (`scrfd_500m` or `scrfd_2.5g`) | ONNX Runtime |
| Recognition | InsightFace `buffalo_sc` (MobileFaceNet/ArcFace family) | ONNX Runtime |
| Quantization | INT8 dynamic where practical | onnxruntime |

- Embedding size: **512** floats (float32 in Redis).
- Comparison: **cosine similarity** only (1:1 against the selected `face_id`).
- Client should downscale toward ~640×640 before upload; server also enforces `MAX_IMAGE_SIDE`.

Latency targets (guidance, not hard SLOs in MVP): detect 30–60 ms, embed 20–40 ms, compare &lt;1 ms, total ~60–120 ms.

## 6. Storage

- **Redis** key: `face:{face_id}` → binary float32[512] (or equivalent compact encoding).
- Optional metadata hash later (`created_at`); not required for MVP.
- No image persistence on disk in the verify/enroll hot path.

## 7. Configuration (environment)

| Variable | Purpose | Default |
|----------|---------|---------|
| `REDIS_URL` | Redis connection | `redis://localhost:6379/0` |
| `FACE_MATCH_THRESHOLD` | Cosine match cutoff | `0.40` |
| `MODEL_DIR` | Directory with ONNX weights | `./models` |
| `MAX_IMAGE_SIDE` | Max image side after resize | `640` |
| `MAX_UPLOAD_BYTES` | Upload size limit | e.g. `5_000_000` |

## 8. Project layout

```
iM.FaceId/
  app/
    main.py           # FastAPI app, OpenAPI/Swagger
    config.py         # pydantic-settings
    api/routes.py     # /v1/faces, /health
    pipeline/         # SCRFD + embedding
    store/redis.py    # face_id ↔ vector
    matching/         # cosine + threshold
  models/             # ONNX files (download script; large binaries gitignored)
  tests/
  docs/superpowers/specs/
  pyproject.toml / requirements.txt
  README.md
  docker-compose.yml  # api + redis for local test
```

## 9. Dependencies (to pin at implement time)

- `fastapi`, `uvicorn[standard]`, `python-multipart`
- `onnxruntime`, `opencv-python-headless`, `numpy`
- `redis` (async or sync client — choose one consistently)
- `pydantic-settings`
- Test: `pytest`, `httpx`

Model assets: SCRFD + `buffalo_sc` ONNX, fetched via a documented script (not committed if large).

## 10. Local run and Swagger test

1. Start Redis (`docker compose up redis` or equivalent).
2. Place/download models into `MODEL_DIR`.
3. `uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`
4. Open `http://127.0.0.1:8000/docs`
5. Call **enroll** with image A → copy `face_id`
6. Call **verify** with same `face_id` and image B (same/different person) → inspect `match` / `score`

## 11. Testing methodology

| Layer | What |
|-------|------|
| Unit | Cosine similarity, threshold boundary, vector serialize/deserialize |
| API | httpx + TestClient; mock pipeline for route/status-code tests |
| Smoke | Real (or fixture) ONNX if available: one same-person pair, one different-person pair |
| Manual | Swagger Try it out as primary developer check for MVP |

## 12. Security (MVP)

- Intended for **internal** network use only.
- No auth in MVP; add API key or mTLS before internet exposure.
- Do not write uploaded images to durable storage in the hot path.

## 13. Deferred (post-MVP)

- Liveness / Silent-Face-Anti-Spoofing (separate layer; must not dominate match latency)
- Queue/workers for high concurrency
- Parallel MyId vs iM.FaceId comparison and cutover metrics
- `DELETE /v1/faces/{face_id}`, listing, bulk enrollment of 200k students
- Integration into existing attendance/finance backend

## 14. Migration note (future)

When integrating with the 200k-student system: keep MyId in parallel for a period; compare agree/disagree and false-reject rates; cut over only after thresholds are validated. Not part of this MVP delivery.
