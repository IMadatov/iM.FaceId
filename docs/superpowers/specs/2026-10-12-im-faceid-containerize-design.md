# iM.FaceId — Docker Compose Containerization (Design)

**Date:** 2026-10-12  
**Status:** Draft (pending written-spec review)  
**Project:** iM.FaceId  
**Parent:** [2026-10-05-im-faceid-webapi-design.md](./2026-10-05-im-faceid-webapi-design.md)

## 1. Intent

Containerize the FaceID FastAPI service so a single server/PC can run the full stack with Docker Compose, with explicit CPU and memory limits per service.

Today only Qdrant runs in Compose; the API runs on the host via `venv` + `uvicorn`. After this change, `docker compose up --build -d` starts API + Qdrant together.

**Success looks like:** one host, one compose file, baked-in models, conservative resource caps, no multi-instance scaling yet.

## 2. Goals and success criteria

| Goal | Success |
|------|---------|
| Full stack in containers | API + Qdrant start via Compose without host venv |
| Resource control | API and Qdrant have documented CPU/RAM limits |
| Reproducible runtime | Models baked into the API image; same image runs on any Docker host |
| Simple ops | Documented `build` / `up` / Swagger smoke path in README |
| Non-goals (this phase) | Multi-instance `--scale`, reverse proxy, Swarm/K8s, CI registry push |

## 3. Constraints (agreed)

| Decision | Choice |
|----------|--------|
| Target environment | Single server/PC, Docker Compose only |
| Model distribution | Bake into API image at build time |
| Scaling | Single API instance; limits only (no proxy / `--scale`) |
| Default budgets | API: 2 CPU / 4 GB; Qdrant: 1 CPU / 2 GB |

## 4. Architecture

```
[Client / Swagger]
        │ :8000
        ▼
┌─────────────────┐     http://qdrant:6333      ┌──────────────┐
│  api (FastAPI)  │ ──────────────────────────► │    qdrant    │
│  models baked   │                             │  volume data │
│  2 CPU / 4 GB   │                             │  1 CPU / 2GB │
└─────────────────┘                             └──────────────┘
```

- **api** — custom image from repo `Dockerfile`; InsightFace + optional liveness ONNX inside the image; talks to Qdrant on the Compose network.
- **qdrant** — existing `qdrant/qdrant` image; durable storage via named volume `qdrant_data`.
- **External ports:** `8000` (API). Qdrant `6333` may remain published for local debug; treat as internal-only in production notes.
- **Stateless API:** no API volume required; vectors live only in Qdrant.

## 5. Components and files

| File | Role |
|------|------|
| `Dockerfile` | Python 3.11 base → system libs for OpenCV/ONNX → `pip install -r requirements.txt` → copy app + download scripts → bake models into `MODEL_DIR` → run `uvicorn app.main:app --host 0.0.0.0 --port 8000` |
| `docker-compose.yml` | `api` + `qdrant` services; resource limits; env; `depends_on`; restart policy |
| `.dockerignore` | Exclude `.venv`, `.git`, `tests`, `graphify-out`, `.worktrees`, caches — keep build context small |
| `README.md` | Replace/extend local-only runbook with Compose-first path |

### 5.1 API image build flow

1. Install OS packages required by InsightFace / OpenCV / ONNX Runtime (CPU).
2. Install Python dependencies from `requirements.txt`.
3. Copy application source and `scripts/`.
4. At **build time**, run model download scripts so weights land under `MODEL_DIR=/app/models`.
5. Set env defaults: `MODEL_DIR=/app/models`, `LIVENESS_MODEL_PATH=/app/models/liveness/minifasnet_v2.onnx`.
6. Expose `8000`; CMD runs uvicorn.

Build failure if model download fails — do not ship an image without required weights.

### 5.2 Compose resource limits

| Service | CPUs | Memory |
|---------|------|--------|
| `api` | `2.0` | `4g` |
| `qdrant` | `1.0` | `2g` |

Use Compose fields supported by the local Docker Engine (e.g. `cpus` / `mem_limit`, or equivalent `deploy.resources` where applicable). Values are defaults; operators may override via compose edit or env-driven substitution if added later.

### 5.3 Restart and dependency

- `restart: unless-stopped` on both services.
- `api` `depends_on` `qdrant`. Prefer a Qdrant healthcheck so API starts after Qdrant accepts HTTP; if healthcheck is impractical in the first cut, document short connection retry / restart behavior.

## 6. Configuration and data flow

### 6.1 Startup

1. Operator runs `docker compose up --build -d`.
2. Qdrant becomes healthy; API starts and connects to `http://qdrant:6333`.
3. Client uses `http://<host>:8000/docs` for manual tests (enroll / verify / search unchanged).

### 6.2 Environment

Compose sets at least:

| Variable | Container value |
|----------|-----------------|
| `QDRANT_URL` | `http://qdrant:6333` |
| `QDRANT_COLLECTION` | `faces` (default) |
| `MODEL_DIR` | `/app/models` |
| `LIVENESS_MODEL_PATH` | `/app/models/liveness/minifasnet_v2.onnx` |

Existing threshold / search / groups env vars keep current defaults; no API contract changes.

### 6.3 Persistence

- **Durable:** Qdrant named volume only.
- **Ephemeral:** API filesystem (models are in the image layers).

## 7. Error handling

| Case | Behavior |
|------|----------|
| Model bake fails at build | Image build fails; fix network/scripts before deploy |
| Qdrant not ready at API start | Healthcheck + restart, or short connect retry; API must not silently succeed without store |
| OOM / CPU throttle | Docker enforces limits; request latency may rise; no app-level change required |
| Missing liveness model at runtime | Same as today: liveness requests → `503` if path absent (bake should include it if scripts succeed) |

## 8. Testing and verification

| Check | How |
|-------|-----|
| Image builds | `docker compose build` |
| Stack runs | `docker compose up -d` |
| API reachable | Open `/docs` on port 8000 |
| Smoke | Swagger enroll (+ optional verify) against live stack |
| Unit tests | Remain host-side `pytest` for this phase; full container E2E not required |

## 9. Out of scope (deferred)

- `docker compose up --scale api=N` and load balancer / nginx / Traefik
- Multi-host Swarm or Kubernetes
- Publishing images to a registry / CI pipeline
- Baking alternative “slim” images without models (volume-mounted models)
- Changing FaceID API routes or matching thresholds

## 10. Implementation notes

- Prefer extending the existing `docker-compose.yml` rather than introducing a second prod-only file in this phase.
- Keep host `venv` + uvicorn path in README as optional for development, but document Compose as the supported single-server deploy path.
- After implementation, revisit model layer caching (multi-stage / separate download layer) only if rebuild time becomes painful.
