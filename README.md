# iM.FaceId

iM.FaceId is an on-premise, CPU-only FaceID microservice for internal student attendance verification. It enrolls face embeddings in Qdrant, supports 1:1 verify and 1:N search for duplicate detection, and exposes a FastAPI OpenAPI surface at `/docs` for manual testing. MVP is intended for internal network use only; there is no authentication layer.

## Prerequisites

- Docker and Docker Compose (API + Qdrant)
- Optional for host-side development: Python 3.11+

## Deploy with Docker Compose (recommended)

Build the API image (downloads InsightFace + liveness models into the image) and start the stack:

```bash
docker compose up --build -d
```

**Daily restarts** (no download): omit `--build` so Docker reuses the existing image:

```bash
docker compose up -d
```

Rebuild only when `requirements.txt` / download scripts change, or after a cache wipe. Model download is a separate Docker layer, so editing `app/` code does **not** re-download models.

- API / Swagger: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- Qdrant HTTP/gRPC are bound to loopback only (`127.0.0.1:6333` / `6334`) for local debug; the API reaches Qdrant on the Compose network

Resource defaults (edit `docker-compose.yml` to change):

| Service | CPUs | Memory |
|---------|------|--------|
| `api` | 4.0 | 4g |
| `qdrant` | 1.0 | 2g |

Postgres is **not** started by Compose — the API uses the host OS Postgres instance. Copy `.env.example` → `.env` (gitignored) with `DATABASE_URL` pointing at `host.docker.internal:5432` for Compose, or `localhost:5432` for host uvicorn. Ensure DB/user exist and `pg_hba.conf` allows Docker bridge connections.

Stop:

```bash
docker compose down
```

## Local setup (optional host venv)

Use this only when iterating on Python code without rebuilding the image.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Start Qdrant only (Postgres must already be running on the host):

```bash
docker compose up -d qdrant
```

Download InsightFace models (SCRFD + `buffalo_sc`):

```bash
python scripts/download_models.py
```

Optional liveness / anti-spoof model (MiniFASNet-V2 ONNX):

```bash
python scripts/download_liveness_model.py
```

InsightFace writes weights under `MODEL_DIR` (default `./models`). The library typically stores the pack at `<MODEL_DIR>/models/buffalo_sc` — set `MODEL_DIR` before running the script if you want a different location. Liveness defaults to `./models/liveness/minifasnet_v2.onnx`.

Run the API on the host:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## Swagger manual checklist

Use **Try it out** on each endpoint in order:

1. **Enroll** — `POST /v1/faces` with JSON body (`image_base64` + optional knobs) → save the returned `face_id`. Set `"liveness": true` to run MiniFASNet anti-spoof first.
2. **Update** — `PUT /v1/faces/{face_id}` with the same JSON shape → same `face_id`, `"updated": true`.
3. **Search** — `POST /v1/faces/search` with JSON (`image_base64`, optional `limit` / `min_score` / liveness knobs) → top-K similar `face_id`s and scores.
4. **Groups rebuild** — `POST /v1/faces/groups/rebuild` → starts an async ANN-graph job (`202` + `run_id`). List runs with `GET /v1/faces/groups/rebuild` (newest `started_at` first) or poll `GET /v1/faces/groups/rebuild/{run_id}` until `completed`. Concurrent rebuilds return `409`.
5. **Groups** — `GET /v1/faces/groups` → reads the latest **completed** run from Postgres (not on-demand O(n²)). Empty if no rebuild has finished yet.
6. **Verify** — `POST /v1/faces/{face_id}/verify` with JSON (`image_base64`, optional `match_threshold` / liveness knobs) → `match` and `score` vs threshold.
7. **Delete** — `DELETE /v1/faces/{face_id}` → `"deleted": true`; verify/search should no longer find that id (`404` or absent from results).
8. **Clear all** — `DELETE /v1/faces` → wipes every embedding (`deleted_count`); use only from a trusted backend (no auth in MVP). Group tables in Postgres are unchanged until the next rebuild.
9. **Count / list / exists / stats** — `GET /v1/faces/count`, `GET /v1/faces?limit=&offset=`, `GET /v1/faces/{face_id}`, `GET /v1/stats` (no vectors returned).

Image endpoints use **`application/json`** (Swagger Request body like a normal login API). Required: `image_base64` (raw base64 or `data:image/...;base64,...`). Optional: `liveness` (default `false`), `blur_min_variance`, `brightness_min`, `brightness_max`, `min_side_px`, `liveness_threshold`; verify also `match_threshold`; search also `limit` / `min_score`. Omit a key → env default. Full override — clients may relax or disable checks (same disable values as env).

Liveness is **off by default**. When `liveness` is `true`, the service detects the face first, runs **face quality** checks, then crops with `LIVENESS_CROP_SCALE` (Silent-Face 2.7) and runs MiniFASNet. Quality or spoof fail → `422` (distinct messages); if the ONNX model is missing → `503`.

```bash
IMG=$(base64 -w0 face.png)
curl -s http://localhost:8000/v1/faces \
  -H 'Content-Type: application/json' \
  -d "{\"image_base64\":\"$IMG\",\"liveness\":true,\"blur_min_variance\":80,\"liveness_threshold\":0.7}"
```

Optional: enroll a second photo of the same person, then **rebuild groups** or **Search** — both ids should appear together (fraud signal).

## Environment variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `QDRANT_URL` | Qdrant HTTP URL | `http://localhost:6333` (Compose: `http://qdrant:6333`) |
| `QDRANT_COLLECTION` | Collection name | `faces` |
| `DATABASE_URL` | SQLAlchemy URL for group runs (host Postgres) | See `.env` / `.env.example` (`imfaceid` @ `localhost` or `host.docker.internal`) |
| `FACE_MATCH_THRESHOLD` | Default 1:1 verify cutoff (JSON `match_threshold` overrides) | `0.40` |
| `FACE_SEARCH_MIN_SCORE` | Default min score for search / group rebuild (JSON `min_score` on search; Query on rebuild) | `0.40` |
| `FACE_SEARCH_DEFAULT_LIMIT` | Default top-K | `5` |
| `FACE_SEARCH_MAX_LIMIT` | Maximum search limit | `20` |
| `FACE_GROUPS_MIN_SIZE` | Default minimum group size | `2` |
| `FACE_GROUPS_ANN_TOP_K` | ANN neighbor edges per face during rebuild | `20` |
| `FACE_GROUPS_KEEP_RUNS` | Completed runs retained after prune | `2` |
| `FACE_GROUPS_MAX_FACES` | Legacy soft-warning threshold on rebuild size | `5000` |
| `FACE_LIST_DEFAULT_LIMIT` | Default page size for `GET /v1/faces` | `100` |
| `FACE_LIST_MAX_LIMIT` | Max page size for face id listing | `1000` |
| `MODEL_DIR` | ONNX / InsightFace weights directory | `./models` (Compose image: `/app/models`) |
| `LIVENESS_MODEL_PATH` | MiniFASNet-V2 ONNX path | `./models/liveness/minifasnet_v2.onnx` (Compose: `/app/models/liveness/minifasnet_v2.onnx`) |
| `LIVENESS_THRESHOLD` | Default min live-class probability (JSON `liveness_threshold` overrides) | `0.50` |
| `LIVENESS_CROP_SCALE` | Face crop expansion for MiniFASNet | `2.7` |
| `FACE_BLUR_MIN_VARIANCE` | Default min Laplacian variance (`liveness=true`); JSON override; `0` disables | `50` |
| `FACE_BRIGHTNESS_MIN` | Default min mean gray; JSON override; `0` disables low check | `40` |
| `FACE_BRIGHTNESS_MAX` | Default max mean gray; JSON override; `255` disables high check | `220` |
| `FACE_MIN_SIDE_PX` | Default min face bbox side in px; JSON override; `0` disables | `80` |
| `MAX_IMAGE_SIDE` | Max image side after resize | `640` |
| `MAX_UPLOAD_BYTES` | Upload size limit (bytes) | `5000000` |

You can place these in a `.env` file in the project root (see `app/config.py`).

## Security (MVP)

- **Internal network only** — do not expose this service to the public internet without auth (API key, mTLS, etc.).
- **No auth in MVP** — callers are trusted on the internal network.
- Uploaded images are not durably stored by the service; only embeddings live in Qdrant. Note that Starlette may spool large multipart uploads (over ~1 MB) to temporary files, which are deleted when the request ends.

## Tests

```bash
pytest tests/ -v -k "not insightface_smoke"
```

Set `RUN_INSIGHTFACE_SMOKE=1` to include the optional InsightFace integration smoke test (requires downloaded models).
