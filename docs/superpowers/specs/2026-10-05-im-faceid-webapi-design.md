# iM.FaceId — On-Premise CPU FaceID Web API (Design)

**Date:** 2026-10-05  
**Status:** Approved (conversational design; revised for Qdrant + search)  
**Project:** iM.FaceId

## 1. Intent

Build an internal FaceID microservice that replaces external MyId calls for student attendance verification. MVP supports:

1. **Enroll** — image → embedding → generated `face_id` stored in a vector DB  
2. **Update** — replace embedding (reference image) for an existing `face_id`  
3. **Delete** — remove a `face_id` and its vector from Qdrant  
4. **Verify** — `face_id` + image → 1:1 cosine match  
5. **Search** — image → top-K similar faces from the DB (detect one person enrolled under multiple IDs / proxy attendance fraud)

**Why vector DB:** Later (and in MVP via search) we must find look-alike faces across the corpus, not only compare against one known ID.

**Out of scope for MVP:** teacher mobile app changes, financial/attendance business logic, liveness/anti-spoofing, request queue workers, MyId parallel comparison, automatic enroll blocking on duplicates (warning via search only).

## 2. Goals and success criteria

| Goal | Success |
|------|---------|
| Independent of MyId for face match | Enroll + update + delete + verify on local CPU + Qdrant |
| Duplicate / look-alike discovery | `POST /v1/faces/search` returns top-K similar `face_id`s with scores |
| Fast enough for classroom verify | Target ~60–120 ms per verify (best-effort in MVP) |
| Easy manual test | OpenAPI Swagger UI at `/docs` with Try it out |
| Durable embeddings | Vectors survive API restart (Qdrant persistence) |

## 3. Architecture

Single FastAPI service (`im-faceid`) with layered modules:

```
Client (Swagger / future backend)
    → API (routes, validation)
    → Pipeline (detect → embed via ONNX)
    → Store (Qdrant: point id = face_id, vector = 512-d)
    → Matching (1:1 retrieve + cosine; 1:N ANN search)
```

**Approach (approved):** Minimal FastAPI — inference in-process; embeddings in **Qdrant** (cosine distance). No Celery/RQ in MVP. Redis is **not** used for vectors.

## 4. API

Base path: `/v1`. Interactive docs: `/docs` (Swagger), `/redoc`.

### `POST /v1/faces`

- **Request:** `multipart/form-data` field `image` (file).
- **Behavior:** Validate → resize ≤ `MAX_IMAGE_SIDE` → detect → 512-d embed → upsert point in Qdrant with new UUID as point id → discard image bytes.
- **Response:** `201 Created`
  ```json
  { "face_id": "<uuid>" }
  ```

### `PUT /v1/faces/{face_id}`

- **Request:** path `face_id`, `multipart/form-data` field `image`.
- **Behavior:** Require existing point → same pipeline → **overwrite** vector for that point id. `face_id` unchanged.
- **Response:** `200 OK`
  ```json
  { "face_id": "<uuid>", "updated": true }
  ```
- **Errors:** `404` if missing; same image errors as enroll.

### `DELETE /v1/faces/{face_id}`

- **Request:** path `face_id` only.
- **Behavior:** Delete the Qdrant point for that id (vector + payload). Idempotent preference: if already missing, return `404` (explicit) so clients know the id was not present.
- **Response:** `200 OK`
  ```json
  { "face_id": "<uuid>", "deleted": true }
  ```
- **Errors:** `404` if `face_id` not found; `503` if Qdrant unavailable.

### `POST /v1/faces/{face_id}/verify`

- **Request:** path `face_id`, `multipart/form-data` field `image`.
- **Behavior:** Retrieve vector by `face_id` → embed probe → cosine similarity → `match = score >= FACE_MATCH_THRESHOLD`.
- **Response:** `200 OK`
  ```json
  {
    "face_id": "<uuid>",
    "match": true,
    "score": 0.72,
    "threshold": 0.40
  }
  ```

### `POST /v1/faces/search`

- **Request:** `multipart/form-data`:
  - `image` (required)
  - `limit` (optional query or form, default `5`, max e.g. `20`)
  - `min_score` (optional, default = `FACE_MATCH_THRESHOLD` or a dedicated `FACE_SEARCH_MIN_SCORE`)
- **Behavior:** Embed probe → Qdrant ANN search (cosine) → return matches at/above `min_score`, excluding nothing by default (caller may ignore self after enroll).
- **Response:** `200 OK`
  ```json
  {
    "results": [
      { "face_id": "<uuid>", "score": 0.91 },
      { "face_id": "<uuid>", "score": 0.88 }
    ],
    "threshold": 0.40
  }
  ```
- **Fraud use case:** If the same physical person was enrolled twice, search on a live photo (or on one enrollment image) surfaces multiple high-scoring `face_id`s for investigation. MVP does **not** auto-reject enroll; clients call search when needed.

### `GET /health`

- **Response:** `200` when models loaded and Qdrant is reachable; otherwise `503` with reason.

### Error responses

| Condition | Status |
|-----------|--------|
| Invalid/oversized file | `400` |
| Unknown `face_id` | `404` |
| No face / multiple faces | `422` |
| Model or Qdrant unavailable | `503` |

## 5. Model stack (CPU)

| Stage | Model | Runtime |
|-------|--------|---------|
| Detection | SCRFD lightweight (`scrfd_500m` or `scrfd_2.5g`) | ONNX Runtime |
| Recognition | InsightFace `buffalo_sc` (MobileFaceNet/ArcFace family) | ONNX Runtime |
| Quantization | INT8 dynamic where practical | onnxruntime |

- Embedding size: **512** floats (`float32`).
- Verify: **1:1** cosine against retrieved vector.
- Search: **1:N** ANN cosine in Qdrant.
- Client should downscale toward ~640×640; server enforces `MAX_IMAGE_SIDE`.

Latency targets (guidance): detect 30–60 ms, embed 20–40 ms, 1:1 compare <1 ms, ANN search typically low tens of ms at ~200k (hardware-dependent). Verify total ~60–120 ms.

## 6. Storage (Qdrant)

- **Collection:** e.g. `faces`
- **Distance:** Cosine
- **Point id:** `face_id` (UUID)
- **Vector:** 512-d float32
- **Payload (optional MVP):** `created_at` / `updated_at` timestamps
- Update = overwrite vector for the same point id
- Delete = remove point by id
- No image persistence on disk in the hot path (only vectors in Qdrant)
- Local: Qdrant via `docker compose` with persistent volume

## 7. Configuration (environment)

| Variable | Purpose | Default |
|----------|---------|---------|
| `QDRANT_URL` | Qdrant HTTP URL | `http://localhost:6333` |
| `QDRANT_COLLECTION` | Collection name | `faces` |
| `FACE_MATCH_THRESHOLD` | 1:1 verify cutoff | `0.40` |
| `FACE_SEARCH_MIN_SCORE` | Default min score for search | `0.40` |
| `FACE_SEARCH_DEFAULT_LIMIT` | Default top-K | `5` |
| `MODEL_DIR` | ONNX weights directory | `./models` |
| `MAX_IMAGE_SIDE` | Max image side after resize | `640` |
| `MAX_UPLOAD_BYTES` | Upload size limit | e.g. `5_000_000` |

## 8. Project layout

```
iM.FaceId/
  app/
    main.py              # FastAPI app, OpenAPI/Swagger
    config.py            # pydantic-settings
    api/routes.py        # /v1/faces, /health
    pipeline/            # SCRFD + embedding
    store/qdrant.py      # upsert, get, search
    matching/            # cosine helpers / threshold
  models/                # ONNX files (download script; gitignore large binaries)
  tests/
  docs/superpowers/specs/
  pyproject.toml / requirements.txt
  README.md
  docker-compose.yml     # api + qdrant for local test
```

## 9. Dependencies (to pin at implement time)

- `fastapi`, `uvicorn[standard]`, `python-multipart`
- `onnxruntime`, `opencv-python-headless`, `numpy`
- `qdrant-client`
- `pydantic-settings`
- Test: `pytest`, `httpx`

Model assets: SCRFD + `buffalo_sc` ONNX, fetched via a documented script (not committed if large).

## 10. Local run and Swagger test

1. Start Qdrant (`docker compose up qdrant` or full stack).
2. Place/download models into `MODEL_DIR`.
3. `uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`
4. Open `http://127.0.0.1:8000/docs`
5. **Enroll** image A → `face_id_1`
6. **Enroll** same person image A′ (optional) → `face_id_2`
7. **Search** with a live/same-person photo → expect both IDs with high scores (fraud signal)
8. **Update** one id with a newer photo → same `face_id`
9. **Verify** `face_id_1` + probe → `match` / `score`
10. **Delete** a test `face_id` → confirm verify/search no longer returns it (`404` / absent from results)

## 11. Testing methodology

| Layer | What |
|-------|------|
| Unit | Cosine/threshold; Qdrant store mocked |
| API | TestClient: enroll/update/delete/verify/search status codes; update/delete/verify `404` |
| Smoke | Two enrollments of same face → search returns both above min_score; different person below or ranked lower |
| Manual | Swagger Try it out as primary MVP check |

## 12. Security (MVP)

- Intended for **internal** network use only.
- No auth in MVP; add API key or mTLS before internet exposure.
- Do not write uploaded images to durable storage in the hot path.
- Search results are sensitive (can link identities); keep API internal.

## 13. Deferred (post-MVP)

- Auto-reject or `409` on enroll when search finds a near-duplicate (option C)
- Liveness / Silent-Face-Anti-Spoofing
- Queue/workers for high concurrency
- Parallel MyId vs iM.FaceId comparison and cutover metrics
- Listing endpoints, bulk enrollment of 200k students
- Integration into existing attendance/finance backend

## 14. Migration note (future)

When integrating with the 200k-student system: keep MyId in parallel for a period; compare agree/disagree and false-reject rates; use search to audit duplicate enrollments; cut over only after thresholds are validated. Not part of this MVP delivery.
