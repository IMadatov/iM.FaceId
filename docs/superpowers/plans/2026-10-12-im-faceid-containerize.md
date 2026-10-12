# iM.FaceId Docker Compose Containerization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the FaceID FastAPI service and Qdrant together via Docker Compose on a single host, with models baked into the API image and conservative CPU/RAM limits.

**Architecture:** Extend the existing Compose file with an `api` service built from a new `Dockerfile`. Build-time scripts download InsightFace + MiniFASNet into `/app/models`. Qdrant keeps the named volume; API is stateless and points at `http://qdrant:6333`. No scale, proxy, or registry in this phase.

**Tech Stack:** Docker, Docker Compose, Python 3.11, FastAPI/uvicorn (existing), InsightFace/ONNX (existing), Qdrant `v1.19.1`, pytest for file-contract tests.

**Spec:** `docs/superpowers/specs/2026-10-12-im-faceid-containerize-design.md`

## Global Constraints

- Target: **single server/PC**, Docker Compose only — no Swarm/K8s
- Models: **bake into API image** at build time (`MODEL_DIR=/app/models`)
- Scaling: **single API instance**; resource limits only (no `--scale`, no reverse proxy)
- Default budgets: API **2 CPU / 4 GB**; Qdrant **1 CPU / 2 GB**
- Liveness path: `LIVENESS_MODEL_PATH=/app/models/liveness/minifasnet_v2.onnx`
- `QDRANT_URL` inside Compose: `http://qdrant:6333` (not localhost)
- Extend existing `docker-compose.yml` — do not add a second prod-only compose file
- No API route or threshold changes
- Host `venv` + uvicorn remains documented as optional local-dev path
- TDD: failing contract test → implement → pass → commit per task

## Review Focus

- Qdrant still starting when API lifespan runs → API must wait on healthy Qdrant (or restart), never serve without store
- Compose leaves `QDRANT_URL=http://localhost:6333` → API cannot reach Qdrant on the Compose network
- Build context includes `.venv` / `models/` / `.git` → bloated or broken builds
- Image builds without running download scripts → runtime lifespan fails for missing InsightFace weights
- Resource limits omitted or only on one service → operator cannot enforce the agreed CPU/RAM budgets

---

## File map

| Path | Responsibility |
|------|----------------|
| `.dockerignore` | Exclude local junk from build context |
| `Dockerfile` | API image: deps, bake models, uvicorn CMD |
| `docker-compose.yml` | `api` + `qdrant`, limits, healthcheck, env |
| `README.md` | Compose-first deploy; keep optional host-venv path |
| `tests/test_docker_contract.py` | File-contract tests for ignore/Dockerfile/compose |
| `docs/superpowers/specs/2026-10-12-im-faceid-containerize-design.md` | Mark status Approved after ship (last task) |

---

### Task 1: Dockerignore and compose/Dockerfile contract tests

**Files:**
- Create: `.dockerignore`
- Create: `tests/test_docker_contract.py`
- Modify: (none yet for compose/Dockerfile — tests must fail first)

**Interfaces:**
- Consumes: nothing
- Produces: pytest module asserting required strings/keys in `.dockerignore`, `Dockerfile`, and `docker-compose.yml` so later tasks have a fixed contract

- [ ] **Step 1: Write failing contract tests**

```python
# tests/test_docker_contract.py
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_dockerignore_excludes_heavy_local_paths():
    text = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    for pattern in (".venv", ".git", "tests", "graphify-out", ".worktrees", "models"):
        assert pattern in text, f"missing {pattern!r} in .dockerignore"


def test_dockerfile_bakes_models_and_runs_uvicorn():
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "MODEL_DIR=/app/models" in text
    assert "scripts/download_models.py" in text
    assert "scripts/download_liveness_model.py" in text
    assert "uvicorn" in text
    assert "0.0.0.0" in text
    assert "8000" in text


def test_compose_api_and_qdrant_limits_and_env():
    text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "api:" in text
    assert "qdrant:" in text
    assert "mem_limit: 4g" in text
    assert 'cpus: "2.0"' in text or "cpus: '2.0'" in text
    assert "mem_limit: 2g" in text
    assert 'cpus: "1.0"' in text or "cpus: '1.0'" in text
    assert "QDRANT_URL=http://qdrant:6333" in text or "QDRANT_URL: http://qdrant:6333" in text
    assert "MODEL_DIR=/app/models" in text or "MODEL_DIR: /app/models" in text
    assert "LIVENESS_MODEL_PATH=/app/models/liveness/minifasnet_v2.onnx" in text or (
        "LIVENESS_MODEL_PATH: /app/models/liveness/minifasnet_v2.onnx" in text
    )
    assert "condition: service_healthy" in text
    assert "restart: unless-stopped" in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_docker_contract.py -v`  
Expected: FAIL (missing `.dockerignore` and/or `Dockerfile`; compose lacks `api` / limits)

- [ ] **Step 3: Create `.dockerignore`**

```gitignore
.git
.venv
.worktrees
.pytest_cache
__pycache__
*.pyc
.env
.idea
.vscode
graphify-out
tests
docs
models
*.onnx
*.md
!requirements.txt
```

Note: `docs` and `*.md` are safe to ignore for the runtime image; do **not** ignore `requirements.txt`, `app/`, or `scripts/`.

- [ ] **Step 4: Run only the dockerignore test (expect still fail on Dockerfile/compose)**

Run: `pytest tests/test_docker_contract.py::test_dockerignore_excludes_heavy_local_paths -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add .dockerignore tests/test_docker_contract.py
git commit -m "$(cat <<'EOF'
test: add Docker contract tests and .dockerignore

Pin Compose/Dockerfile expectations before adding the API image.
EOF
)"
```

---

### Task 2: API Dockerfile with baked models

**Files:**
- Create: `Dockerfile`
- Test: `tests/test_docker_contract.py::test_dockerfile_bakes_models_and_runs_uvicorn`

**Interfaces:**
- Consumes: `requirements.txt`, `app/`, `scripts/download_models.py`, `scripts/download_liveness_model.py` (existing)
- Produces: image build recipe that sets `MODEL_DIR=/app/models`, bakes both model sets, runs uvicorn on `:8000`

- [ ] **Step 1: Confirm Dockerfile test still fails**

Run: `pytest tests/test_docker_contract.py::test_dockerfile_bakes_models_and_runs_uvicorn -v`  
Expected: FAIL with missing `Dockerfile` (or incomplete content)

- [ ] **Step 2: Create `Dockerfile`**

```dockerfile
FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts

ENV MODEL_DIR=/app/models \
    LIVENESS_MODEL_PATH=/app/models/liveness/minifasnet_v2.onnx \
    PYTHONUNBUFFERED=1

RUN python scripts/download_models.py \
    && python scripts/download_liveness_model.py

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

If `insightface` needs a compiler on the target arch, add `build-essential` / `g++` to the `apt-get install` line and keep them only if required for the wheel install (prefer wheels; do not add packages “just in case” after a clean build succeeds without them).

- [ ] **Step 3: Run Dockerfile contract test**

Run: `pytest tests/test_docker_contract.py::test_dockerfile_bakes_models_and_runs_uvicorn -v`  
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add Dockerfile
git commit -m "$(cat <<'EOF'
feat: add Dockerfile that bakes FaceID models into the API image

Install CPU runtime deps and download InsightFace plus MiniFASNet at build time.
EOF
)"
```

---

### Task 3: Compose `api` service, limits, and Qdrant healthcheck

**Files:**
- Modify: `docker-compose.yml` (replace entire file with the content below)
- Test: `tests/test_docker_contract.py::test_compose_api_and_qdrant_limits_and_env`

**Interfaces:**
- Consumes: `Dockerfile` from Task 2; existing Qdrant volume name `qdrant_data`
- Produces: Compose stack where `api` waits for healthy `qdrant`, uses baked model paths, and enforces CPU/RAM limits

- [ ] **Step 1: Confirm compose contract test fails on current file**

Run: `pytest tests/test_docker_contract.py::test_compose_api_and_qdrant_limits_and_env -v`  
Expected: FAIL (no `api` service / limits / health condition)

- [ ] **Step 2: Replace `docker-compose.yml` with**

```yaml
services:
  qdrant:
    image: qdrant/qdrant:v1.19.1
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage
    restart: unless-stopped
    cpus: "1.0"
    mem_limit: 2g
    healthcheck:
      test:
        [
          "CMD-SHELL",
          "bash -c 'exec 3<>/dev/tcp/127.0.0.1/6333'",
        ]
      interval: 5s
      timeout: 3s
      retries: 10
      start_period: 5s

  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      QDRANT_URL: http://qdrant:6333
      QDRANT_COLLECTION: faces
      MODEL_DIR: /app/models
      LIVENESS_MODEL_PATH: /app/models/liveness/minifasnet_v2.onnx
    depends_on:
      qdrant:
        condition: service_healthy
    restart: unless-stopped
    cpus: "2.0"
    mem_limit: 4g

volumes:
  qdrant_data:
```

If the Qdrant image lacks bash/`/dev/tcp`, switch the healthcheck `test` to whatever HTTP probe the image supports (e.g. `wget`/`curl` against `http://127.0.0.1:6333/readyz`) while keeping `condition: service_healthy` and the same retry timing intent. Update the contract test only if the literal `condition: service_healthy` string remains (it must).

- [ ] **Step 3: Run compose contract test**

Run: `pytest tests/test_docker_contract.py::test_compose_api_and_qdrant_limits_and_env -v`  
Expected: PASS

- [ ] **Step 4: Validate Compose file with Docker**

Run: `docker compose config`  
Expected: exits 0; rendered config includes `api` and `qdrant`

- [ ] **Step 5: Run full contract suite**

Run: `pytest tests/test_docker_contract.py -v`  
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add docker-compose.yml
git commit -m "$(cat <<'EOF'
feat: add Compose API service with CPU/RAM limits and Qdrant healthcheck

Wire QDRANT_URL to the Compose DNS name and gate API start on healthy Qdrant.
EOF
)"
```

---

### Task 4: README Compose-first runbook and live smoke

**Files:**
- Modify: `README.md` (Prerequisites + Local setup sections; keep Swagger checklist)
- Test: live commands below (manual/smoke; unit suite still host-side)

**Interfaces:**
- Consumes: Task 2 image + Task 3 compose
- Produces: documented `docker compose up --build -d` path; verified `/health` and `/docs` on port 8000

- [ ] **Step 1: Update README Prerequisites and lead with Compose deploy**

Replace the **Prerequisites** and **Local setup** sections (through the uvicorn run block) with:

```markdown
## Prerequisites

- Docker and Docker Compose (API + Qdrant)
- Optional for host-side development: Python 3.11+

## Deploy with Docker Compose (recommended)

Build the API image (downloads InsightFace + liveness models into the image) and start the stack:

```bash
docker compose up --build -d
```

- API / Swagger: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- Qdrant HTTP is published on `6333` for local debug; treat as internal-only on a shared server

Resource defaults (edit `docker-compose.yml` to change):

| Service | CPUs | Memory |
|---------|------|--------|
| `api` | 2.0 | 4g |
| `qdrant` | 1.0 | 2g |

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

Start Qdrant only:

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
```

Keep the existing **Swagger manual checklist**, **Environment variables**, **Security**, and **Tests** sections. In the environment table, note that Compose overrides `QDRANT_URL`, `MODEL_DIR`, and `LIVENESS_MODEL_PATH` as above.

- [ ] **Step 2: Build and start the stack**

Run:

```bash
docker compose up --build -d
```

Expected: build succeeds (model downloads during `RUN`); both containers running; `docker compose ps` shows `api` and `qdrant` healthy/up.

- [ ] **Step 3: Smoke health and docs**

Run:

```bash
curl -sf http://127.0.0.1:8000/health
curl -sf -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/docs
```

Expected: health JSON includes `"status":"ok"`; docs returns `200`.

- [ ] **Step 4: Optional Swagger enroll smoke**

Use `/docs` **Try it out** on `POST /v1/faces` with a clear single-face photo, or skip if no sample image is available. Expected: `201` with a `face_id` when an image is provided.

- [ ] **Step 5: Run unit tests (host) to ensure no regressions**

Run: `pytest -q`  
Expected: PASS (including `tests/test_docker_contract.py`)

- [ ] **Step 6: Commit**

```bash
git add README.md
git commit -m "$(cat <<'EOF'
docs: make Docker Compose the primary FaceID deploy path

Document resource limits, health URL, and keep host venv as optional.
EOF
)"
```

---

### Task 5: Mark design spec approved

**Files:**
- Modify: `docs/superpowers/specs/2026-10-12-im-faceid-containerize-design.md` (status line only)

**Interfaces:**
- Consumes: shipped Tasks 1–4
- Produces: spec status reflects implementation readiness closure

- [ ] **Step 1: Update status header**

Change:

```markdown
**Status:** Draft (pending written-spec review)
```

to:

```markdown
**Status:** Approved (implemented via 2026-10-12 plan)
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-10-12-im-faceid-containerize-design.md
git commit -m "$(cat <<'EOF'
docs: mark containerization design spec approved

Point status at the 2026-10-12 implementation plan after Compose ship.
EOF
)"
```

---

## Self-review (plan author)

| Spec requirement | Task |
|------------------|------|
| Full stack in containers | Task 3 + Task 4 smoke |
| CPU/RAM limits | Task 3 + contract test |
| Models baked into image | Task 2 |
| Simple ops / README | Task 4 |
| Single compose file | Task 3 |
| Qdrant health / API wait | Task 3 `depends_on` + healthcheck |
| No scale/proxy/K8s | Out of scope; not in tasks |
| Unit tests host-side | Task 1–3 contract tests; Task 4 `pytest -q` |

Placeholder scan: none. Type/name consistency: `MODEL_DIR=/app/models`, `LIVENESS_MODEL_PATH=/app/models/liveness/minifasnet_v2.onnx`, `QDRANT_URL=http://qdrant:6333` used uniformly. Review Focus items mapped to Task 1/3 contract assertions and Task 3 healthcheck + Task 4 `/health` smoke.
