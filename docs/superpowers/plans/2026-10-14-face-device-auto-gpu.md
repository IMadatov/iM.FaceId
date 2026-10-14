# Face Device Auto GPU Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve `FACE_DEVICE=auto|cpu|cuda` at startup so InsightFace and liveness share CUDA when available, with CPU fallback, while keeping default Compose on CPU and adding an optional `gpu` profile.

**Architecture:** A small `resolve_face_device()` helper probes ONNX Runtime providers once. Production lifespan passes the resolved providers/`ctx_id` into InsightFace and liveness. Default `Dockerfile` stays on `onnxruntime`; `Dockerfile.gpu` + Compose profile `gpu` install `onnxruntime-gpu` and expose the NVIDIA device.

**Tech Stack:** FastAPI, InsightFace, ONNX Runtime / ONNX Runtime GPU, Docker Compose profiles, pytest, pydantic-settings.

**Spec:** `docs/superpowers/specs/2026-10-14-face-device-auto-gpu-design.md`

## Global Constraints

- Default `FACE_DEVICE=auto`; also accept `cpu` and `cuda`
- Invalid `FACE_DEVICE` → **fail startup** with clear error (do not silently normalize)
- CUDA missing or init fail → **CPU fallback + warning**; process stays up
- InsightFace and liveness must use the **same** effective device
- Quality unchanged: `buffalo_sc`, `det_size=(640, 640)`, float32, `max_image_side=640`
- Default Compose `api` stays CPU (`Dockerfile` + `onnxruntime`)
- GPU via `docker compose --profile gpu` → service `api-gpu` on port `8000` (do not run `api` and `api-gpu` together)
- Do **not** add device field to `/health`
- No TensorRT, multi-GPU, INT8, or hard-fail-without-fallback mode
- TDD: failing test → implement → pass → commit per task
- Use `.venv/bin/pytest` when system `pytest` is missing

## Review Focus

- Hardcoding `CPUExecutionProvider` left in InsightFace or liveness → GPU path never used
- `onnxruntime` and `onnxruntime-gpu` both in the same requirements file → broken installs
- `api` and `api-gpu` both without profiles / both binding `:8000` by default → port conflict on `compose up`
- CUDA probe only checks env string, never `get_available_providers()` → false CUDA on CPU image
- Pipeline CUDA construct failure not caught → startup crash instead of CPU fallback
- Production wiring mocks still assume old `__init__` signatures → false failures

---

## File map

| Path | Responsibility |
|------|----------------|
| `app/pipeline/device.py` | Resolve requested device → providers, ctx_id, fallback reason |
| `app/config.py` | `face_device` setting |
| `app/pipeline/onnx_insightface.py` | Accept providers + ctx_id |
| `app/pipeline/onnx_liveness.py` | Accept providers; pass through `load_liveness_checker` |
| `app/main.py` | Resolve once; construct pipelines; log; CUDA→CPU rebuild fallback |
| `requirements-gpu.txt` | GPU deps (`onnxruntime-gpu` instead of `onnxruntime`) |
| `Dockerfile.gpu` | CUDA runtime image + bake models |
| `docker-compose.yml` | Add `api-gpu` under profile `gpu` |
| `.env.example` | Document `FACE_DEVICE` |
| `README.md` | GPU profile ops notes |
| `tests/test_device_resolve.py` | Unit tests for resolver |
| `tests/test_docker_contract.py` | GPU compose/Dockerfile contract |
| `tests/test_production_wiring.py` | Update mocks for new kwargs |
| `docs/superpowers/specs/2026-10-14-face-device-auto-gpu-design.md` | Mark implemented after ship |

---

### Task 1: Device resolver (TDD)

**Files:**
- Create: `app/pipeline/device.py`
- Create: `tests/test_device_resolve.py`
- Modify: (none)

**Interfaces:**
- Consumes: `onnxruntime.get_available_providers` (injected/mockable via optional `available_providers` arg for tests)
- Produces:
  - `@dataclass(frozen=True) class ResolvedDevice` with fields: `requested: str`, `effective: str` (`"cpu"`|`"cuda"`), `providers: list[str]`, `ctx_id: int`, `fallback_reason: str | None`
  - `def resolve_face_device(requested: str, *, available_providers: list[str] | None = None) -> ResolvedDevice`
  - Valid requested values: `auto`, `cpu`, `cuda` (case-sensitive lowercase as stored in settings)
  - Invalid requested → `ValueError` with message containing `FACE_DEVICE`
  - CUDA preferred providers: `["CUDAExecutionProvider", "CPUExecutionProvider"]`, `ctx_id=0`
  - CPU providers: `["CPUExecutionProvider"]`, `ctx_id=-1`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_device_resolve.py
import pytest

from app.pipeline.device import resolve_face_device


def test_cpu_always_cpu():
    r = resolve_face_device("cpu", available_providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert r.effective == "cpu"
    assert r.providers == ["CPUExecutionProvider"]
    assert r.ctx_id == -1
    assert r.fallback_reason is None
    assert r.requested == "cpu"


def test_auto_without_cuda_uses_cpu():
    r = resolve_face_device("auto", available_providers=["CPUExecutionProvider"])
    assert r.effective == "cpu"
    assert r.providers == ["CPUExecutionProvider"]
    assert r.ctx_id == -1
    assert r.fallback_reason is not None


def test_auto_with_cuda_uses_cuda():
    r = resolve_face_device(
        "auto", available_providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    assert r.effective == "cuda"
    assert r.providers[0] == "CUDAExecutionProvider"
    assert "CPUExecutionProvider" in r.providers
    assert r.ctx_id == 0
    assert r.fallback_reason is None


def test_cuda_missing_falls_back_to_cpu():
    r = resolve_face_device("cuda", available_providers=["CPUExecutionProvider"])
    assert r.effective == "cpu"
    assert r.ctx_id == -1
    assert r.fallback_reason is not None


def test_invalid_device_raises():
    with pytest.raises(ValueError, match="FACE_DEVICE"):
        resolve_face_device("gpu", available_providers=["CPUExecutionProvider"])
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `.venv/bin/pytest tests/test_device_resolve.py -v`  
Expected: FAIL (import error / module missing)

- [ ] **Step 3: Implement resolver**

```python
# app/pipeline/device.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResolvedDevice:
    requested: str
    effective: str
    providers: list[str]
    ctx_id: int
    fallback_reason: str | None = None


_CPU = "CPUExecutionProvider"
_CUDA = "CUDAExecutionProvider"
_VALID = frozenset({"auto", "cpu", "cuda"})


def resolve_face_device(
    requested: str, *, available_providers: list[str] | None = None
) -> ResolvedDevice:
    if requested not in _VALID:
        raise ValueError(
            f"Invalid FACE_DEVICE={requested!r}; expected one of: auto, cpu, cuda"
        )
    if available_providers is None:
        import onnxruntime as ort

        available_providers = list(ort.get_available_providers())

    cuda_ok = _CUDA in available_providers

    if requested == "cpu" or (requested in {"auto", "cuda"} and not cuda_ok):
        reason = None
        if requested != "cpu" and not cuda_ok:
            reason = f"{_CUDA} not available in {available_providers}"
        return ResolvedDevice(
            requested=requested,
            effective="cpu",
            providers=[_CPU],
            ctx_id=-1,
            fallback_reason=reason,
        )

    return ResolvedDevice(
        requested=requested,
        effective="cuda",
        providers=[_CUDA, _CPU],
        ctx_id=0,
        fallback_reason=None,
    )
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `.venv/bin/pytest tests/test_device_resolve.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/pipeline/device.py tests/test_device_resolve.py
git commit -m "$(cat <<'EOF'
feat: add FACE_DEVICE resolver with CUDA auto-fallback

EOF
)"
```

---

### Task 2: Settings + pipeline constructors accept resolved device

**Files:**
- Modify: `app/config.py`
- Modify: `app/pipeline/onnx_insightface.py`
- Modify: `app/pipeline/onnx_liveness.py`
- Modify: `tests/test_insightface_pipeline_unit.py` (only if constructor kwargs break stubs)
- Modify: `tests/test_onnx_liveness_unit.py` (pass providers if needed)

**Interfaces:**
- Consumes: `ResolvedDevice` from Task 1 (or `providers` / `ctx_id` kwargs)
- Produces:
  - `Settings.face_device: str = "auto"`
  - `InsightFacePipeline.__init__(self, model_dir, model_name="buffalo_sc", embedding_dim=512, *, providers: list[str] | None = None, ctx_id: int = -1)`
  - Defaults: `providers=None` → `["CPUExecutionProvider"]` (keeps unit tests / download script behavior)
  - `OnnxLivenessChecker.__init__(..., *, providers: list[str] | None = None)`
  - `load_liveness_checker(model_path, *, crop_scale=2.7, providers: list[str] | None = None)`

- [ ] **Step 1: Add failing test for default providers still CPU when omitted**

Extend `tests/test_device_resolve.py` is already done; for pipelines, add a focused unit assertion via monkeypatch in a new test file or extend existing:

```python
# tests/test_pipeline_device_kwargs.py
from app.pipeline.onnx_insightface import InsightFacePipeline


def test_insightface_passes_providers_and_ctx(monkeypatch):
    captured = {}

    class FakeFA:
        def __init__(self, name, root, providers):
            captured["providers"] = providers

        def prepare(self, ctx_id, det_size):
            captured["ctx_id"] = ctx_id
            captured["det_size"] = det_size

    monkeypatch.setattr(
        "insightface.app.FaceAnalysis",
        FakeFA,
        raising=False,
    )
    # Patch where FaceAnalysis is imported inside __init__
    import app.pipeline.onnx_insightface as mod

    monkeypatch.setattr(mod, "FaceAnalysis", FakeFA, raising=False)

    # InsightFacePipeline imports FaceAnalysis inside __init__; monkeypatch insightface.app
    import insightface.app as ia

    monkeypatch.setattr(ia, "FaceAnalysis", FakeFA)

    InsightFacePipeline(
        "/tmp",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        ctx_id=0,
    )
    assert captured["providers"][0] == "CUDAExecutionProvider"
    assert captured["ctx_id"] == 0
```

Note: because `FaceAnalysis` is imported inside `__init__` with `from insightface.app import FaceAnalysis`, monkeypatch `insightface.app.FaceAnalysis` **before** constructing the pipeline (as above with `ia.FaceAnalysis`).

- [ ] **Step 2: Run test — expect FAIL** (unexpected keyword `providers`)

Run: `.venv/bin/pytest tests/test_pipeline_device_kwargs.py -v`  
Expected: FAIL

- [ ] **Step 3: Update config + pipelines**

`app/config.py` — add field:

```python
face_device: str = "auto"
```

`app/pipeline/onnx_insightface.py` — change `__init__` to:

```python
def __init__(
    self,
    model_dir: str,
    model_name: str = "buffalo_sc",
    embedding_dim: int = 512,
    *,
    providers: list[str] | None = None,
    ctx_id: int = -1,
) -> None:
    from insightface.app import FaceAnalysis

    self._dim = embedding_dim
    self._app = FaceAnalysis(
        name=model_name,
        root=model_dir,
        providers=providers or ["CPUExecutionProvider"],
    )
    self._app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    self._ready = True
```

`app/pipeline/onnx_liveness.py` — change session creation:

```python
def __init__(
    self,
    model_path: str,
    *,
    crop_scale: float = 2.7,
    providers: list[str] | None = None,
) -> None:
    import onnxruntime as ort

    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"liveness model not found: {model_path}")
    self._session = ort.InferenceSession(
        str(path), providers=providers or ["CPUExecutionProvider"]
    )
    # ... rest unchanged
```

```python
def load_liveness_checker(
    model_path: str, *, crop_scale: float = 2.7, providers: list[str] | None = None
):
    from app.pipeline.fake_liveness import UnavailableLivenessChecker

    try:
        return OnnxLivenessChecker(
            model_path, crop_scale=crop_scale, providers=providers
        )
    except Exception:
        return UnavailableLivenessChecker()
```

- [ ] **Step 4: Run unit tests**

Run: `.venv/bin/pytest tests/test_pipeline_device_kwargs.py tests/test_insightface_pipeline_unit.py tests/test_onnx_liveness_unit.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/config.py app/pipeline/onnx_insightface.py app/pipeline/onnx_liveness.py tests/test_pipeline_device_kwargs.py
git commit -m "$(cat <<'EOF'
feat: pass ONNX providers and ctx_id into face pipelines

EOF
)"
```

---

### Task 3: Production lifespan wires resolver + CUDA construct fallback

**Files:**
- Modify: `app/main.py` (`_production_lifespan`)
- Modify: `tests/test_production_wiring.py`

**Interfaces:**
- Consumes: `resolve_face_device`, `Settings.face_device`, pipeline kwargs from Task 2
- Produces: lifespan that:
  1. Calls `resolve_face_device(settings.face_device)`
  2. Logs `face device requested=%s effective=%s fallback=%s`
  3. Builds InsightFace with `providers=` / `ctx_id=`
  4. If construction raises and `effective == "cuda"`, log warning, resolve/force CPU providers, rebuild once
  5. Passes same final providers into `load_liveness_checker(..., providers=...)`
  6. Invalid `face_device` → `ValueError` propagates; wrap as `RuntimeError` with clear message if needed for TestClient parity

Helper recommended in `app/main.py` or `app/pipeline/device.py`:

```python
def build_insightface_with_fallback(
    settings, resolved: ResolvedDevice
) -> tuple[InsightFacePipeline, ResolvedDevice]:
    try:
        pipe = InsightFacePipeline(
            settings.model_dir,
            embedding_dim=settings.embedding_dim,
            providers=resolved.providers,
            ctx_id=resolved.ctx_id,
        )
        return pipe, resolved
    except Exception as exc:
        if resolved.effective != "cuda":
            raise
        logger.warning(
            "CUDA InsightFace init failed (%s); falling back to CPU", exc
        )
        cpu = ResolvedDevice(
            requested=resolved.requested,
            effective="cpu",
            providers=["CPUExecutionProvider"],
            ctx_id=-1,
            fallback_reason=f"InsightFace CUDA init failed: {exc}",
        )
        pipe = InsightFacePipeline(
            settings.model_dir,
            embedding_dim=settings.embedding_dim,
            providers=cpu.providers,
            ctx_id=cpu.ctx_id,
        )
        return pipe, cpu
```

- [ ] **Step 1: Update production wiring test to accept new kwargs**

In `tests/test_production_wiring.py`, change the InsightFace mock to accept `**kwargs`:

```python
monkeypatch.setattr(
    "app.pipeline.onnx_insightface.InsightFacePipeline",
    lambda model_dir, **kwargs: sentinel_pipe,
)
```

Add a new test that invalid device fails startup:

```python
def test_production_startup_fails_on_invalid_face_device(monkeypatch):
    from app.config import Settings

    monkeypatch.setattr(
        "app.config.get_settings",
        lambda: Settings(face_device="gpu", model_dir="."),
    )
    # Still need store/db mocks if resolve happens after those — ensure resolve runs early
    app = create_app(testing=False)
    with pytest.raises((RuntimeError, ValueError), match="FACE_DEVICE"):
        with TestClient(app):
            pass
```

Implement resolve **before** heavy Qdrant/DB work so invalid device fails fast. Order in lifespan:

1. `settings = get_settings()`
2. `resolved = resolve_face_device(settings.face_device)` (catch `ValueError` → `RuntimeError`)
3. log
4. InsightFace with fallback helper
5. Qdrant / DB (existing)
6. liveness with `providers=final_resolved.providers`

- [ ] **Step 2: Run new/updated tests — expect FAIL until lifespan wired**

Run: `.venv/bin/pytest tests/test_production_wiring.py -v`  
Expected: FAIL on invalid device test until implemented

- [ ] **Step 3: Wire `_production_lifespan`**

Implement the order and helper above. Keep existing error wrapping for InsightFace/Qdrant/DB.

- [ ] **Step 4: Run wiring + device tests**

Run: `.venv/bin/pytest tests/test_production_wiring.py tests/test_device_resolve.py tests/test_pipeline_device_kwargs.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/main.py tests/test_production_wiring.py
git commit -m "$(cat <<'EOF'
feat: wire FACE_DEVICE into production startup with CUDA fallback

EOF
)"
```

---

### Task 4: GPU Docker artifacts + Compose profile (contract TDD)

**Files:**
- Create: `requirements-gpu.txt`
- Create: `Dockerfile.gpu`
- Modify: `docker-compose.yml`
- Modify: `tests/test_docker_contract.py`

**Interfaces:**
- Consumes: existing CPU Dockerfile patterns (model bake scripts, uvicorn CMD)
- Produces: GPU image build recipe + `api-gpu` service with `profiles: ["gpu"]`

- [ ] **Step 1: Extend contract tests (failing)**

Add to `tests/test_docker_contract.py`:

```python
def test_compose_gpu_profile_api_gpu_service():
    text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "api-gpu:" in text
    assert "profiles:" in text
    assert "gpu" in text
    assert "Dockerfile.gpu" in text
    assert "FACE_DEVICE" in text
    assert "gpus:" in text or "nvidia" in text.lower()


def test_dockerfile_gpu_uses_onnxruntime_gpu():
    text = (ROOT / "Dockerfile.gpu").read_text(encoding="utf-8")
    assert "requirements-gpu.txt" in text
    req = (ROOT / "requirements-gpu.txt").read_text(encoding="utf-8")
    assert "onnxruntime-gpu" in req
    assert "onnxruntime>=" not in req.replace("onnxruntime-gpu", "")
    assert "download_models.py" in text
    assert "uvicorn" in text
```

- [ ] **Step 2: Run — expect FAIL**

Run: `.venv/bin/pytest tests/test_docker_contract.py::test_compose_gpu_profile_api_gpu_service tests/test_docker_contract.py::test_dockerfile_gpu_uses_onnxruntime_gpu -v`  
Expected: FAIL

- [ ] **Step 3: Create `requirements-gpu.txt`**

Copy `requirements.txt` and replace the `onnxruntime>=...` line with:

```text
onnxruntime-gpu>=1.19.0
```

Keep all other pins identical to `requirements.txt`.

- [ ] **Step 4: Create `Dockerfile.gpu`**

Use an NVIDIA CUDA runtime base compatible with ORT GPU wheels (pin at implement time; starting point):

```dockerfile
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /app

COPY requirements-gpu.txt .
RUN pip install --no-cache-dir -r requirements-gpu.txt

ENV MODEL_DIR=/app/models \
    LIVENESS_MODEL_PATH=/app/models/liveness/minifasnet_v2.onnx \
    FACE_DEVICE=auto \
    PYTHONUNBUFFERED=1

COPY scripts/download_models.py scripts/download_liveness_model.py ./scripts/
RUN python scripts/download_models.py \
    && python scripts/download_liveness_model.py

COPY app ./app
COPY scripts ./scripts

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

If the chosen CUDA tag is unavailable on the build host, adjust to the nearest available `nvidia/cuda:*-cudnn-runtime-*` tag that installs a working `onnxruntime-gpu` wheel — document the final tag in the Dockerfile comment.

- [ ] **Step 5: Add `api-gpu` to `docker-compose.yml`**

```yaml
  api-gpu:
    profiles: ["gpu"]
    build:
      context: .
      dockerfile: Dockerfile.gpu
    ports:
      - "8000:8000"
    env_file:
      - .env
    extra_hosts:
      - "host.docker.internal:host-gateway"
    environment:
      QDRANT_URL: http://qdrant:6333
      QDRANT_COLLECTION: faces
      MODEL_DIR: /app/models
      LIVENESS_MODEL_PATH: /app/models/liveness/minifasnet_v2.onnx
      FACE_DEVICE: auto
    depends_on:
      qdrant:
        condition: service_healthy
    restart: unless-stopped
    gpus: all
    cpus: "2.0"
    mem_limit: 4g
```

Keep existing `api` service unchanged (CPU, no profile).

- [ ] **Step 6: Run contract tests — expect PASS**

Run: `.venv/bin/pytest tests/test_docker_contract.py -v`  
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add requirements-gpu.txt Dockerfile.gpu docker-compose.yml tests/test_docker_contract.py
git commit -m "$(cat <<'EOF'
feat: add Compose gpu profile and Dockerfile.gpu

EOF
)"
```

---

### Task 5: Docs + env example + spec status

**Files:**
- Modify: `.env.example`
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-10-14-face-device-auto-gpu-design.md` (status line)

**Interfaces:**
- Consumes: Task 4 service name / commands
- Produces: operator docs for CPU vs GPU deploy

- [ ] **Step 1: Update `.env.example`**

Append:

```bash
# Face inference device: auto (CUDA if available else CPU), cpu, or cuda
FACE_DEVICE=auto
```

- [ ] **Step 2: Update README**

- Soften intro “CPU-only” → on-premise FaceID with CPU default and optional GPU profile
- After the CPU deploy section, add:

```markdown
## GPU deploy (optional)

Requires NVIDIA driver + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on the host.

Do **not** run default `api` and `api-gpu` together (both bind host port `8000`).

```bash
docker compose --profile gpu up --build -d
```

`FACE_DEVICE` defaults to `auto` (CUDA when available, otherwise CPU fallback).

| Service | CPUs | Memory | Notes |
|---------|------|--------|-------|
| `api-gpu` | 2.0 | 4g | needs NVIDIA GPU |
| `qdrant` | 1.0 | 2g | same as CPU stack |
```

- [ ] **Step 3: Mark spec status**

In the design doc header:  
`**Status:** Approved (implemented via 2026-10-14 plan)`

- [ ] **Step 4: Run full relevant suite**

Run: `.venv/bin/pytest tests/test_device_resolve.py tests/test_pipeline_device_kwargs.py tests/test_production_wiring.py tests/test_docker_contract.py tests/test_insightface_pipeline_unit.py tests/test_onnx_liveness_unit.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add .env.example README.md docs/superpowers/specs/2026-10-14-face-device-auto-gpu-design.md
git commit -m "$(cat <<'EOF'
docs: document FACE_DEVICE auto GPU Compose profile

EOF
)"
```

---

## Spec coverage checklist (self-review)

| Spec requirement | Task |
|------------------|------|
| `FACE_DEVICE=auto\|cpu\|cuda`, default auto | 1, 2 |
| Shared device InsightFace + liveness | 3 |
| CUDA fail → CPU + warning | 1, 3 |
| Default Compose CPU unchanged | 4 (`api` untouched) |
| Profile `gpu` → `api-gpu` + `Dockerfile.gpu` + `onnxruntime-gpu` | 4 |
| Quality defaults unchanged | 2 (det_size 640), no INT8 tasks |
| No `/health` device field | — intentionally omitted |
| Unit + contract tests | 1, 4 |
| README + `.env.example` | 5 |
| Invalid device fails startup | 1, 3 |

## Placeholder / consistency notes

- `ResolvedDevice` field names are stable across Tasks 1–3: `requested`, `effective`, `providers`, `ctx_id`, `fallback_reason`
- Pipeline kwargs: `providers`, `ctx_id` (InsightFace); `providers` (liveness)
- Compose service name is exactly `api-gpu`; profile name exactly `gpu`
