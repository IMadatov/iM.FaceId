# iM.FaceId — Dynamic CPU/GPU Face Device (Design)

**Date:** 2026-10-14  
**Status:** Approved (pending implementation plan)  
**Project:** iM.FaceId  
**Parent:** [2026-10-05-im-faceid-webapi-design.md](./2026-10-05-im-faceid-webapi-design.md)  
**Related:** [2026-10-12-im-faceid-containerize-design.md](./2026-10-12-im-faceid-containerize-design.md)

## 1. Intent

Allow the FaceID API to run face detection/embedding and liveness inference on **GPU when available**, while keeping the existing **CPU-only Compose path** as the default and unchanged for hosts without NVIDIA hardware.

Operators select behavior via `FACE_DEVICE=auto` (default). CUDA availability is probed at startup; if CUDA is missing or fails to initialize, the service **falls back to CPU** with a warning log and continues serving.

**Success looks like:** one codebase; default `docker compose up` still CPU; optional `docker compose --profile gpu up` uses a GPU image and RTX-class hardware when present; quality settings (float32, `det_size` 640, `buffalo_sc`) unchanged.

## 2. Goals and success criteria

| Goal | Success |
|------|---------|
| Dynamic device | `FACE_DEVICE=auto\|cpu\|cuda`; default `auto` |
| Shared device for both models | InsightFace and MiniFASNet liveness use the same resolved device |
| Safe fallback | CUDA unavailable/fail → CPU + warning; process stays up |
| CPU deploy unchanged | Default Compose `api` service keeps CPU `Dockerfile` + `onnxruntime` |
| Optional GPU deploy | Compose profile `gpu` runs `api-gpu` with `Dockerfile.gpu` + `onnxruntime-gpu` + GPU device access |
| Quality preserved | No INT8, no smaller `det_size`, same model names/dims |
| Non-goals | TensorRT, multi-GPU, `/health` device field, INT8, model swap, forcing GPU-or-die without fallback |

## 3. Constraints (agreed)

| Decision | Choice |
|----------|--------|
| Device selection | **A** — `auto` (CUDA if available, else CPU) |
| Docker packaging | **B** — two profiles: default CPU; `--profile gpu` for GPU |
| CUDA failure | **A** — fall back to CPU + warning (do not crash) |
| Implementation approach | Shared config + device resolver; both pipelines consume resolve result |
| Explicit `cpu` / `cuda` | Supported; `cuda` with failed init still falls back to CPU (same as `auto` outcome; log distinguishes requested vs effective) |
| Port conflict | GPU profile exposes `api-gpu` on `:8000`; do not run default `api` and `api-gpu` together on the same host port |

## 4. Architecture

```
FACE_DEVICE (auto|cpu|cuda)
        │
        ▼
┌───────────────────────┐
│  resolve_face_device  │ ── probe CUDAExecutionProvider / session smoke
└───────────────────────┘
        │
        ├─ effective=cuda → providers=[CUDA, CPU], ctx_id=0
        └─ effective=cpu  → providers=[CPU],       ctx_id=-1
                │
        ┌───────┴────────┐
        ▼                ▼
 InsightFacePipeline   OnnxLivenessChecker
 (detect + embed)      (MiniFASNet)
```

Startup (production lifespan):

1. Load settings (`face_device`).
2. Call resolver once; log `requested=` and `effective=`.
3. Construct InsightFace + liveness with resolved providers / `ctx_id`.
4. Serve requests as today (no per-request device switching).

### 4.1 Resolve rules

| `FACE_DEVICE` | Behavior |
|---------------|----------|
| `auto` | Prefer CUDA if probe succeeds; else CPU |
| `cpu` | Always CPU (skip CUDA probe) |
| `cuda` | Prefer CUDA; on probe/init failure → CPU + warning |
| other | Treat as invalid config at settings validation (reject or normalize to `auto` — implementation must pick **reject with clear error** at startup) |

Probe must be cheap and deterministic: check that `CUDAExecutionProvider` is in `onnxruntime.get_available_providers()` and optionally open a trivial session or rely on FaceAnalysis/liveness construction catching failure and falling back before marking the app ready. Preferred: resolve providers first; if pipeline construction with CUDA fails, rebuild once with CPU and log warning.

## 5. Components and files

| File | Role |
|------|------|
| `app/config.py` | Add `face_device: str = "auto"` (`FACE_DEVICE`) |
| `app/pipeline/device.py` | New: `ResolvedDevice` dataclass + `resolve_face_device(requested: str) -> ResolvedDevice` |
| `app/pipeline/onnx_insightface.py` | Accept providers / `ctx_id` (or `ResolvedDevice`); remove hardcoded CPU-only |
| `app/pipeline/onnx_liveness.py` | Same provider wiring |
| `app/main.py` | Resolve once in production lifespan; pass into both loaders; log effective device |
| `.env.example` | Document `FACE_DEVICE=auto` |
| `requirements.txt` | Unchanged default: `onnxruntime` (CPU) |
| `requirements-gpu.txt` (new) | Same pins as CPU requirements except `onnxruntime-gpu` instead of `onnxruntime` |
| `Dockerfile` | Unchanged CPU path |
| `Dockerfile.gpu` (new) | CUDA-capable base (or NVIDIA CUDA runtime image + Python deps); install `requirements-gpu.txt`; bake models same as CPU Dockerfile |
| `docker-compose.yml` | Keep `api` (CPU). Add `api-gpu` with `profiles: ["gpu"]`, `gpus: all` (or device reservation), `FACE_DEVICE=auto`, build `Dockerfile.gpu` |
| `README.md` | Document GPU profile prerequisites and `docker compose --profile gpu up --build -d` |
| `tests/test_device_resolve.py` (new) | Unit tests for resolve/fallback with mocked ORT providers |
| `tests/test_docker_contract.py` | Assert GPU profile / `Dockerfile.gpu` / `onnxruntime-gpu` contract |

### 5.1 `ResolvedDevice` (conceptual)

- `requested: str` — raw setting
- `effective: Literal["cpu", "cuda"]`
- `providers: list[str]`
- `ctx_id: int` — `0` for cuda, `-1` for cpu
- `fallback_reason: str | None` — set when requested cuda/auto but effective cpu

### 5.2 Compose sketch

- **`api`** — existing CPU service; no profile; port `8000`.
- **`api-gpu`** — `profiles: ["gpu"]`; build `dockerfile: Dockerfile.gpu`; env includes `FACE_DEVICE=auto`; GPU access via Compose `gpus: all` (Docker Engine with NVIDIA Container Toolkit); mem limit similar to API (`4g`); CPU limit may stay modest (e.g. `2.0`–`4.0`) because inference is GPU-bound; depends on healthy `qdrant` like `api`.
- Operators run **either** `api` **or** `api-gpu`, not both (same host port `8000`).

## 6. Error handling and logging

| Case | Behavior |
|------|----------|
| Invalid `FACE_DEVICE` | Fail startup with clear validation error |
| `cpu` | No CUDA attempt; log effective=cpu |
| `auto`/`cuda`, CUDA providers missing | effective=cpu; warning with reason |
| `auto`/`cuda`, provider present but session/model init fails | Catch, retry construction on CPU once; warning; if CPU also fails → fail startup (existing behavior) |
| GPU image on host without toolkit | Fallback to CPU; service still serves (slower) |

Do **not** add device to `/health` in this phase.

## 7. Testing

| Test | Assert |
|------|--------|
| Unit: auto, no CUDA | effective=cpu, CPU provider only |
| Unit: cpu | effective=cpu, no cuda in providers |
| Unit: cuda missing | effective=cpu, fallback_reason set |
| Unit: cuda available (mocked) | effective=cuda, CUDA first in providers, ctx_id=0 |
| Contract: compose | `api-gpu` + `profiles` contains `gpu`; build uses `Dockerfile.gpu` |
| Contract: Dockerfile.gpu | references `requirements-gpu.txt` / `onnxruntime-gpu` |
| Optional smoke | Skip unless CUDA present; assert pipeline `ready()` |

## 8. Ops notes

- Host GPU path requires NVIDIA driver + NVIDIA Container Toolkit compatible with the chosen CUDA base image and `onnxruntime-gpu` wheel.
- RTX 50-series may need a recent CUDA / ORT combo; pin versions in `Dockerfile.gpu` / `requirements-gpu.txt` at implement time after a smoke check on target hardware.
- Quality defaults remain: `buffalo_sc`, `det_size=(640, 640)`, float32, `max_image_side=640`.

## 9. Deferred

- Expose effective device on `/health` or metrics
- TensorRT / FP16 execution providers
- Explicit `FACE_DEVICE=cuda` hard-fail mode (no CPU fallback)
- Single multi-arch image bundling both ORT packages (rejected; pip conflict)
