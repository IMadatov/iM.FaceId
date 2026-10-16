# iM.FaceId — Client API Reference

Base URL (local Compose): `http://127.0.0.1:8000`  
Interactive docs: `http://127.0.0.1:8000/docs`  
Content-Type for bodies: `application/json`  
Auth: none (internal network only)

Unknown JSON fields are rejected (`extra=forbid`) → **422**.

---

## Conventions

### Images

Image endpoints require `image_base64`:

- Raw base64 of PNG/JPEG bytes, **or**
- Data URL: `data:image/png;base64,...`

Decoded size must stay under server `MAX_UPLOAD_BYTES` (default 5 MB). Images are resized so the longest side ≤ `MAX_IMAGE_SIDE` (default 640).

### Shared image request fields

Used by enroll, update, search, verify:

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `image_base64` | string | **required** | Face photo |
| `liveness` | bool | `false` | If `true`, run quality + MiniFASNet before business logic |
| `blur_min_variance` | number \| null | server env | Used only when `liveness=true`; `0` disables blur check |
| `brightness_min` | number \| null | server env | `0` disables low-brightness check |
| `brightness_max` | number \| null | server env | `255` disables high-brightness check |
| `min_side_px` | number \| null | server env | Min face bbox side; `0` disables |
| `liveness_threshold` | number \| null | server env | MiniFASNet live-class cutoff |

Omit a key (or send `null`) → server `.env` default.

### Errors

Response shape: `{ "detail": "<message>" }` (string) or FastAPI validation detail (array/object).

| Status | When |
|--------|------|
| **400** | Bad / empty / oversized / undecodable image; `min_size < 2` on groups |
| **404** | Unknown `face_id` or rebuild `run_id` |
| **409** | Group rebuild already running |
| **422** | No face / multiple faces; quality fail; liveness fail; invalid JSON / unknown fields |
| **503** | Pipeline/store/liveness model unavailable |

Quality vs spoof: both **422**, but `detail` mentions `quality`/`blur`/… vs `liveness`.

### Typical client flow

1. `POST /v1/faces` → store `face_id`
2. Optional: `POST /v1/faces/{id}/verify` or `POST /v1/faces/search`
3. Optional fraud clustering: `POST /v1/faces/groups/rebuild` → poll status → `POST /v1/faces/groups/list`
4. `DELETE /v1/faces/{id}` when done

---

## Endpoints

### `GET /health`

Liveness of API dependencies (models + vector store).

**Response 200**

```json
{ "status": "ok", "detail": null }
```

**503** — pipeline or store not ready.

---

### `GET /v1/stats`

```json
{
  "faces_count": 12,
  "store_ok": true,
  "pipeline_ok": true
}
```

---

### `GET /v1/faces/count`

```json
{ "count": 12 }
```

---

### `POST /v1/faces` — Enroll

Register a new face. Returns a new UUID `face_id`.

**Request**

```json
{
  "image_base64": "<base64>",
  "liveness": false,
  "blur_min_variance": null,
  "brightness_min": null,
  "brightness_max": null,
  "min_side_px": null,
  "liveness_threshold": null
}
```

**Response 201**

```json
{
  "face_id": "550e8400-e29b-41d4-a716-446655440000",
  "liveness_score": null
}
```

`liveness_score` is set only when `liveness` was `true`.

**curl**

```bash
IMG=$(base64 -w0 face.png)
curl -s -X POST "$BASE/v1/faces" \
  -H 'Content-Type: application/json' \
  -d "{\"image_base64\":\"$IMG\",\"liveness\":true}"
```

---

### `PUT /v1/faces/{face_id}` — Update

Replace embedding for an existing face. Same body as enroll.

**Response 200**

```json
{
  "face_id": "550e8400-e29b-41d4-a716-446655440000",
  "updated": true,
  "liveness_score": null
}
```

**404** — face not found.

---

### `POST /v1/faces/{face_id}/verify` — 1:1 verify

Compare probe image to one enrolled face.

**Request** (enroll fields + optional match cutoff)

```json
{
  "image_base64": "<base64>",
  "liveness": false,
  "match_threshold": null
}
```

**Response 200**

```json
{
  "face_id": "550e8400-e29b-41d4-a716-446655440000",
  "match": true,
  "score": 0.87,
  "threshold": 0.4,
  "liveness_score": null
}
```

`match` is `score >= threshold`. `threshold` is the effective value (client override or server default, usually `0.40`).

---

### `POST /v1/faces/search` — 1:N search

Find similar enrolled faces (duplicate / lookalike check).

**Request**

```json
{
  "image_base64": "<base64>",
  "liveness": false,
  "limit": null,
  "min_score": null
}
```

| Field | Notes |
|-------|--------|
| `limit` | Top-K; default `5`, clamped to max `20` |
| `min_score` | Min cosine score; default `0.40` |

**Response 200**

```json
{
  "results": [
    { "face_id": "...", "score": 0.91 },
    { "face_id": "...", "score": 0.55 }
  ],
  "threshold": 0.4,
  "liveness_score": null
}
```

`threshold` echoes the effective `min_score`.

---

### `POST /v1/faces/list` — List face IDs

**Request**

```json
{
  "limit": null,
  "offset": 0
}
```

| Field | Notes |
|-------|--------|
| `limit` | Default `100`, max `1000` |
| `offset` | ≥ `0` |

**Response 200**

```json
{
  "face_ids": ["...", "..."],
  "total": 42,
  "limit": 100,
  "offset": 0
}
```

---

### `GET /v1/faces/{face_id}` — Exists check

**Response 200**

```json
{ "face_id": "...", "exists": true }
```

**404** — not found.

---

### `DELETE /v1/faces/{face_id}` — Delete one

**Response 200**

```json
{ "face_id": "...", "deleted": true }
```

**404** — not found.

---

### `DELETE /v1/faces` — Clear all

Wipes every embedding. No body. Use only from a trusted backend.

**Response 200**

```json
{ "cleared": true, "deleted_count": 42 }
```

Does **not** clear Postgres group-run tables; run a new rebuild after wipe if needed.

---

### `POST /v1/faces/groups/rebuild` — Start group rebuild

Async ANN-graph clustering job. Returns immediately.

**Request**

```json
{
  "min_score": null,
  "min_size": null,
  "top_k": null
}
```

| Field | Notes |
|-------|--------|
| `min_score` | Edge / similarity floor (default `0.40`) |
| `min_size` | Min faces per group (default `2`, must be ≥ `2`) |
| `top_k` | ANN neighbors per face (default `20`, ≥ `1`) |

**Response 202**

```json
{ "run_id": "...", "status": "running" }
```

**409** — another rebuild is already in progress.  
**400** — invalid `min_size` / args.

---

### `GET /v1/faces/groups/rebuild` — List rebuild runs

Newest `started_at` first. No body.

**Response 200**

```json
{
  "runs": [
    {
      "run_id": "...",
      "status": "completed",
      "threshold": 0.4,
      "top_k": 20,
      "min_size": 2,
      "face_count": 100,
      "group_count": 12,
      "error": null,
      "started_at": "2026-10-16T05:00:00",
      "finished_at": "2026-10-16T05:00:05"
    }
  ]
}
```

`status` values include: `pending`, `running`, `completed`, `failed` (and similar).

---

### `GET /v1/faces/groups/rebuild/{run_id}` — Rebuild status

Same object as one element of `runs` above.

**404** — unknown `run_id`.

Poll until `status` is `completed` or `failed`.

---

### `POST /v1/faces/groups/list` — Read groups snapshot

Returns groups from the latest **completed** rebuild (not live O(n²)).

**Request**

```json
{ "min_size": null }
```

`min_size` filters returned groups (default `2`). Values `< 2` → **400**.

**Response 200**

```json
{
  "threshold": 0.4,
  "groups": [
    { "face_ids": ["id-a", "id-b"], "size": 2 }
  ],
  "run_id": "...",
  "built_at": "2026-10-16T05:00:05"
}
```

If no rebuild has completed yet: `groups: []`, `run_id` / `built_at` null.

---

## Pseudocode (client)

```text
function enroll(imageBytes, opts = {}):
  body = { image_base64: base64(imageBytes), ...opts }
  r = HTTP POST /v1/faces  JSON body
  assert r.status == 201
  return r.json.face_id

function verify(faceId, imageBytes, opts = {}):
  r = HTTP POST /v1/faces/{faceId}/verify  JSON { image_base64, ...opts }
  return r.json   # { match, score, threshold, liveness_score }

function search(imageBytes, opts = {}):
  r = HTTP POST /v1/faces/search  JSON { image_base64, ...opts }
  return r.json.results

function rebuildGroups(opts = {}):
  r = HTTP POST /v1/faces/groups/rebuild  JSON opts   # may be {}
  assert r.status == 202
  runId = r.json.run_id
  loop:
    s = HTTP GET /v1/faces/groups/rebuild/{runId}
    if s.json.status in {completed, failed}: return s.json
    sleep(500ms)

function listGroups(opts = {}):
  r = HTTP POST /v1/faces/groups/list  JSON opts     # may be {}
  return r.json.groups
```

---

## Server defaults (reference)

Clients may override per request; these are typical env defaults:

| Setting | Default |
|---------|---------|
| Match threshold | `0.40` |
| Search min score / limit | `0.40` / `5` (max `20`) |
| List limit | `100` (max `1000`) |
| Liveness threshold | `0.50` |
| Blur min variance | `50` |
| Brightness | `40`–`220` |
| Min face side | `80` px |
| Groups min size / ANN top-k | `2` / `20` |

---

## OpenAPI

Prefer generating typed clients from the live schema:

```text
GET http://127.0.0.1:8000/openapi.json
```

Or use Swagger UI **Try it out** at `/docs` while implementing.
