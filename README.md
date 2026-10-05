# iM.FaceId

iM.FaceId is an on-premise, CPU-only FaceID microservice for internal student attendance verification. It enrolls face embeddings in Qdrant, supports 1:1 verify and 1:N search for duplicate detection, and exposes a FastAPI OpenAPI surface at `/docs` for manual testing. MVP is intended for internal network use only; there is no authentication layer.

## Prerequisites

- Python 3.11+
- Docker (for Qdrant)

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Start Qdrant:

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

Run the API:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## Swagger manual checklist

Use **Try it out** on each endpoint in order:

1. **Enroll** — `POST /v1/faces` with a clear single-face photo → save the returned `face_id`. Optional form field `liveness=true` runs MiniFASNet anti-spoof first.
2. **Update** — `PUT /v1/faces/{face_id}` with a newer photo of the same person → same `face_id`, `"updated": true` (optional `liveness=true`).
3. **Search** — `POST /v1/faces/search` with a probe image → top-K similar `face_id`s and scores (useful for duplicate-enrollment checks; optional `liveness=true`).
4. **Groups** — `GET /v1/faces/groups` → on-demand clusters of similar `face_id`s (cosine ≥ threshold, connected components; singletons omitted).
5. **Verify** — `POST /v1/faces/{face_id}/verify` with a probe image → `match` and `score` vs threshold (optional `liveness=true`).
6. **Delete** — `DELETE /v1/faces/{face_id}` → `"deleted": true`; verify/search should no longer find that id (`404` or absent from results).

Liveness is **off by default**. When `liveness=true` and the check fails → `422`; if the ONNX model is missing → `503`.

Optional: enroll a second photo of the same person, then **Groups** or **Search** — both ids should appear together (fraud signal).

## Environment variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `QDRANT_URL` | Qdrant HTTP URL | `http://localhost:6333` |
| `QDRANT_COLLECTION` | Collection name | `faces` |
| `FACE_MATCH_THRESHOLD` | 1:1 verify cutoff | `0.40` |
| `FACE_SEARCH_MIN_SCORE` | Default min score for search | `0.40` |
| `FACE_SEARCH_DEFAULT_LIMIT` | Default top-K | `5` |
| `FACE_SEARCH_MAX_LIMIT` | Maximum search limit | `20` |
| `FACE_GROUPS_MIN_SIZE` | Default minimum group size | `2` |
| `FACE_GROUPS_MAX_FACES` | Max faces allowed for on-demand grouping | `5000` |
| `MODEL_DIR` | ONNX / InsightFace weights directory | `./models` |
| `LIVENESS_MODEL_PATH` | MiniFASNet-V2 ONNX path | `./models/liveness/minifasnet_v2.onnx` |
| `LIVENESS_THRESHOLD` | Min live-class probability | `0.50` |
| `LIVENESS_CROP_SCALE` | Face crop expansion for MiniFASNet | `2.7` |
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
