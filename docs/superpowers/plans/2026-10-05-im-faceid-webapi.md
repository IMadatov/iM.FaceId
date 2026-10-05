# iM.FaceId Web API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a FastAPI FaceID service with enroll/update/delete/verify/search, ONNX CPU embeddings, Qdrant storage, and Swagger at `/docs`.

**Architecture:** Layered FastAPI app — HTTP routes call a face pipeline (detect+embed) and a Qdrant-backed store; verify does 1:1 cosine; search does ANN top-K. Tests use a fake pipeline and a mocked/in-memory store so CI does not need ONNX weights; real models load at runtime for Swagger/manual tests.

**Tech Stack:** Python 3.11+, FastAPI, uvicorn, onnxruntime, opencv-python-headless, numpy, qdrant-client, pydantic-settings, pytest, httpx, Docker Compose (Qdrant).

**Spec:** `docs/superpowers/specs/2026-10-05-im-faceid-webapi-design.md`

## Global Constraints

- Project name: **iM.FaceId**; Python package root: `app/`
- Public API under `/v1`; Swagger `/docs` and ReDoc `/redoc` enabled
- Embedding size: **512** float32; distance: **cosine**
- Vector DB: **Qdrant only** (no Redis for vectors)
- Enroll returns **generated UUID** `face_id` (client does not supply id)
- MVP endpoints: enroll, update, delete, verify, search, health
- Error mapping: `400` bad/oversized file; `404` unknown id; `422` no face / multiple faces; `503` model or Qdrant down
- Defaults: `FACE_MATCH_THRESHOLD=0.40`, `FACE_SEARCH_MIN_SCORE=0.40`, `FACE_SEARCH_DEFAULT_LIMIT=5`, `MAX_IMAGE_SIDE=640`, `MAX_UPLOAD_BYTES=5000000`
- No auth, no liveness, no queue, no auto-reject on duplicate enroll
- Do not persist uploaded images to disk in the hot path
- TDD: failing test → implement → pass → commit per task
- Pin deps in `requirements.txt` at implement time to current stable versions

## Review Focus

- Empty or corrupt image bytes → `400`, never 500
- Image with zero faces or >1 face → `422` with clear detail
- Verify/update/delete unknown UUID → `404`
- Search `limit` above max (20) → clamp or `400` (choose clamp to 20)
- Qdrant connection failure mid-request → `503`, not unhandled exception

---

## File map (create unless noted)

| Path | Responsibility |
|------|----------------|
| `requirements.txt` | Runtime + test deps |
| `pyproject.toml` | pytest/python config |
| `.gitignore` | venv, models binaries, `__pycache__`, `.env` |
| `app/__init__.py` | Package marker |
| `app/config.py` | Settings from env |
| `app/main.py` | FastAPI app factory, lifespan, DI |
| `app/errors.py` | Domain errors → HTTP mapping |
| `app/matching/__init__.py` | Export cosine helpers |
| `app/matching/cosine.py` | Cosine similarity + threshold match |
| `app/imaging.py` | Decode/resize/validate image bytes |
| `app/pipeline/__init__.py` | Exports |
| `app/pipeline/base.py` | `FacePipeline` protocol + domain errors |
| `app/pipeline/fake.py` | Deterministic fake for tests |
| `app/pipeline/onnx_insightface.py` | Real SCRFD+buffalo_sc via InsightFace/ONNX |
| `app/store/__init__.py` | Exports |
| `app/store/base.py` | `FaceStore` protocol |
| `app/store/memory.py` | In-memory store for unit/API tests |
| `app/store/qdrant.py` | Qdrant implementation |
| `app/api/__init__.py` | Exports |
| `app/api/schemas.py` | Pydantic response models |
| `app/api/routes.py` | `/health` + `/v1/faces*` |
| `app/deps.py` | FastAPI dependency getters |
| `scripts/download_models.py` | Ensure InsightFace buffalo_sc models present |
| `docker-compose.yml` | `qdrant` (+ optional `api`) |
| `README.md` | Run + Swagger instructions |
| `tests/conftest.py` | App client with fake pipeline + memory store |
| `tests/test_cosine.py` | Matching unit tests |
| `tests/test_imaging.py` | Image validation tests |
| `tests/test_memory_store.py` | Store contract tests |
| `tests/test_api_faces.py` | HTTP enroll/update/delete/verify/search/health |

---

### Task 1: Project scaffold, settings, and health stub

**Files:**
- Create: `requirements.txt`, `pyproject.toml`, `.gitignore`, `app/__init__.py`, `app/config.py`, `app/main.py`, `app/api/__init__.py`, `app/api/routes.py`, `app/api/schemas.py`, `tests/conftest.py`, `tests/test_health.py`
- Test: `tests/test_health.py`

**Interfaces:**
- Consumes: nothing
- Produces: `Settings` in `app/config.py`; `create_app()` in `app/main.py`; `GET /health` returns JSON

- [ ] **Step 1: Write failing health test**

```python
# tests/test_health.py
from fastapi.testclient import TestClient
from app.main import create_app

def test_health_ok_when_dependencies_ready():
    app = create_app(testing=True)
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
```

- [ ] **Step 2: Run test — expect fail**

Run: `pytest tests/test_health.py::test_health_ok_when_dependencies_ready -v`  
Expected: FAIL (module/app missing)

- [ ] **Step 3: Add dependency files and gitignore**

```text
# requirements.txt
fastapi>=0.115.0
uvicorn[standard]>=0.32.0
python-multipart>=0.0.12
numpy>=1.26.0
opencv-python-headless>=4.10.0
onnxruntime>=1.19.0
insightface>=0.7.3
qdrant-client>=1.12.0
pydantic-settings>=2.6.0
pytest>=8.3.0
httpx>=0.27.0
```

```toml
# pyproject.toml
[project]
name = "im-faceid"
version = "0.1.0"
requires-python = ">=3.11"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

```gitignore
# .gitignore
.venv/
__pycache__/
*.pyc
.pytest_cache/
.env
models/
*.onnx
.idea/
.vscode/
```

- [ ] **Step 4: Implement settings + stub app**

```python
# app/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "faces"
    face_match_threshold: float = 0.40
    face_search_min_score: float = 0.40
    face_search_default_limit: int = 5
    face_search_max_limit: int = 20
    model_dir: str = "./models"
    max_image_side: int = 640
    max_upload_bytes: int = 5_000_000
    embedding_dim: int = 512

def get_settings() -> Settings:
    return Settings()
```

```python
# app/api/schemas.py
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str
    detail: str | None = None

class FaceIdResponse(BaseModel):
    face_id: str

class UpdateResponse(BaseModel):
    face_id: str
    updated: bool = True

class DeleteResponse(BaseModel):
    face_id: str
    deleted: bool = True

class VerifyResponse(BaseModel):
    face_id: str
    match: bool
    score: float
    threshold: float

class SearchHit(BaseModel):
    face_id: str
    score: float

class SearchResponse(BaseModel):
    results: list[SearchHit]
    threshold: float
```

```python
# app/api/routes.py
from fastapi import APIRouter
from app.api.schemas import HealthResponse

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")
```

```python
# app/main.py
from fastapi import FastAPI
from app.api.routes import router

def create_app(*, testing: bool = False) -> FastAPI:
    app = FastAPI(title="iM.FaceId", version="0.1.0")
    app.state.testing = testing
    app.include_router(router)
    return app

app = create_app()
```

```python
# tests/conftest.py
import pytest
from fastapi.testclient import TestClient
from app.main import create_app

@pytest.fixture
def client():
    app = create_app(testing=True)
    with TestClient(app) as c:
        yield c
```

- [ ] **Step 5: Install deps and pass health test**

Run:
```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pytest tests/test_health.py -v
```
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add requirements.txt pyproject.toml .gitignore app tests
git commit -m "chore: scaffold FastAPI app with health endpoint"
```

---

### Task 2: Cosine matching helpers

**Files:**
- Create: `app/matching/__init__.py`, `app/matching/cosine.py`, `tests/test_cosine.py`
- Test: `tests/test_cosine.py`

**Interfaces:**
- Consumes: `numpy`
- Produces:
  - `cosine_similarity(a: np.ndarray, b: np.ndarray) -> float`
  - `is_match(score: float, threshold: float) -> bool`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_cosine.py
import numpy as np
from app.matching.cosine import cosine_similarity, is_match

def test_identical_vectors_score_one():
    v = np.ones(512, dtype=np.float32)
    assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-5)

def test_orthogonal_vectors_score_near_zero():
    a = np.zeros(512, dtype=np.float32); a[0] = 1.0
    b = np.zeros(512, dtype=np.float32); b[1] = 1.0
    assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-5)

def test_is_match_boundary():
    assert is_match(0.40, 0.40) is True
    assert is_match(0.3999, 0.40) is False
```

(Put `import pytest` with the other imports at the top of the file.)

- [ ] **Step 2: Run — expect fail**

Run: `pytest tests/test_cosine.py -v`  
Expected: FAIL import error

- [ ] **Step 3: Implement**

```python
# app/matching/cosine.py
import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32, copy=False).reshape(-1)
    b = b.astype(np.float32, copy=False).reshape(-1)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def is_match(score: float, threshold: float) -> bool:
    return score >= threshold
```

```python
# app/matching/__init__.py
from app.matching.cosine import cosine_similarity, is_match

__all__ = ["cosine_similarity", "is_match"]
```

- [ ] **Step 4: Run — expect pass**

Run: `pytest tests/test_cosine.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/matching tests/test_cosine.py
git commit -m "feat: add cosine similarity helpers"
```

---

### Task 3: Image decode, size limit, resize

**Files:**
- Create: `app/imaging.py`, `app/errors.py`, `tests/test_imaging.py`
- Test: `tests/test_imaging.py`

**Interfaces:**
- Consumes: `Settings.max_image_side`, `Settings.max_upload_bytes`
- Produces:
  - `class InvalidImageError(Exception)`
  - `decode_and_resize(data: bytes, *, max_side: int, max_bytes: int) -> np.ndarray`  # BGR uint8 HWC

- [ ] **Step 1: Write failing tests**

```python
# tests/test_imaging.py
import numpy as np
import cv2
import pytest
from app.imaging import decode_and_resize, InvalidImageError

def _png_bytes(w: int, h: int) -> bytes:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()

def test_rejects_empty_bytes():
    with pytest.raises(InvalidImageError):
        decode_and_resize(b"", max_side=640, max_bytes=5_000_000)

def test_rejects_oversized_upload():
    data = _png_bytes(10, 10)
    with pytest.raises(InvalidImageError):
        decode_and_resize(data, max_side=640, max_bytes=10)

def test_resizes_long_side_to_max():
    data = _png_bytes(1280, 720)
    out = decode_and_resize(data, max_side=640, max_bytes=5_000_000)
    assert max(out.shape[0], out.shape[1]) == 640
```

- [ ] **Step 2: Run — expect fail**

Run: `pytest tests/test_imaging.py -v`  
Expected: FAIL

- [ ] **Step 3: Implement**

```python
# app/errors.py
class InvalidImageError(Exception):
    """Bad or oversized image upload."""

class FaceDetectionError(Exception):
    """No face or multiple faces."""

    def __init__(self, message: str, *, code: str = "face_detection"):
        super().__init__(message)
        self.code = code

class FaceNotFoundError(Exception):
    """Unknown face_id."""

class DependencyUnavailableError(Exception):
    """Model or Qdrant unavailable."""
```

```python
# app/imaging.py
import cv2
import numpy as np
from app.errors import InvalidImageError

def decode_and_resize(data: bytes, *, max_side: int, max_bytes: int) -> np.ndarray:
    if not data:
        raise InvalidImageError("empty image")
    if len(data) > max_bytes:
        raise InvalidImageError("image exceeds max upload size")
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise InvalidImageError("cannot decode image")
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side > max_side:
        scale = max_side / float(long_side)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img
```

Re-export `InvalidImageError` from `app.imaging` for the test import:

```python
# at bottom of app/imaging.py or change tests to import from app.errors
from app.errors import InvalidImageError as InvalidImageError
```

Update `tests/test_imaging.py` imports to:
`from app.errors import InvalidImageError` and `from app.imaging import decode_and_resize`.

- [ ] **Step 4: Run — expect pass**

Run: `pytest tests/test_imaging.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/errors.py app/imaging.py tests/test_imaging.py
git commit -m "feat: validate and resize uploaded images"
```

---

### Task 4: FaceStore protocol + in-memory store

**Files:**
- Create: `app/store/__init__.py`, `app/store/base.py`, `app/store/memory.py`, `tests/test_memory_store.py`
- Test: `tests/test_memory_store.py`

**Interfaces:**
- Consumes: `numpy`, UUID strings
- Produces:
  - `FaceStore` protocol with:
    - `upsert(face_id: str, vector: np.ndarray) -> None`
    - `get(face_id: str) -> np.ndarray | None`
    - `delete(face_id: str) -> bool`  # False if missing
    - `search(vector: np.ndarray, *, limit: int, min_score: float) -> list[tuple[str, float]]`
    - `ping() -> bool`
  - `MemoryFaceStore` implementing the protocol (brute-force cosine for search)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_memory_store.py
import numpy as np
import pytest
from app.store.memory import MemoryFaceStore
from app.matching.cosine import cosine_similarity

def test_upsert_get_delete():
    store = MemoryFaceStore()
    v = np.random.randn(512).astype(np.float32)
    store.upsert("id-1", v)
    got = store.get("id-1")
    assert got is not None
    assert cosine_similarity(got, v) == pytest.approx(1.0, abs=1e-5)
    assert store.delete("id-1") is True
    assert store.get("id-1") is None
    assert store.delete("id-1") is False

def test_search_returns_top_by_score():
    store = MemoryFaceStore()
    base = np.zeros(512, dtype=np.float32); base[0] = 1.0
    near = base.copy(); near[1] = 0.1
    far = np.zeros(512, dtype=np.float32); far[2] = 1.0
    store.upsert("near", near)
    store.upsert("far", far)
    hits = store.search(base, limit=5, min_score=0.0)
    assert hits[0][0] == "near"
```

- [ ] **Step 2: Run — expect fail**

Run: `pytest tests/test_memory_store.py -v`  
Expected: FAIL

- [ ] **Step 3: Implement**

```python
# app/store/base.py
from typing import Protocol
import numpy as np

class FaceStore(Protocol):
    def upsert(self, face_id: str, vector: np.ndarray) -> None: ...
    def get(self, face_id: str) -> np.ndarray | None: ...
    def delete(self, face_id: str) -> bool: ...
    def search(
        self, vector: np.ndarray, *, limit: int, min_score: float
    ) -> list[tuple[str, float]]: ...
    def ping(self) -> bool: ...
```

```python
# app/store/memory.py
import numpy as np
from app.matching.cosine import cosine_similarity

class MemoryFaceStore:
    def __init__(self) -> None:
        self._data: dict[str, np.ndarray] = {}

    def upsert(self, face_id: str, vector: np.ndarray) -> None:
        self._data[face_id] = vector.astype(np.float32, copy=True).reshape(-1)

    def get(self, face_id: str) -> np.ndarray | None:
        v = self._data.get(face_id)
        return None if v is None else v.copy()

    def delete(self, face_id: str) -> bool:
        return self._data.pop(face_id, None) is not None

    def search(
        self, vector: np.ndarray, *, limit: int, min_score: float
    ) -> list[tuple[str, float]]:
        scored = [
            (fid, cosine_similarity(vector, vec))
            for fid, vec in self._data.items()
        ]
        scored = [x for x in scored if x[1] >= min_score]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def ping(self) -> bool:
        return True
```

```python
# app/store/__init__.py
from app.store.memory import MemoryFaceStore

__all__ = ["MemoryFaceStore"]
```

- [ ] **Step 4: Run — expect pass**

Run: `pytest tests/test_memory_store.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/store tests/test_memory_store.py
git commit -m "feat: add FaceStore protocol and memory implementation"
```

---

### Task 5: FacePipeline protocol + fake pipeline

**Files:**
- Create: `app/pipeline/__init__.py`, `app/pipeline/base.py`, `app/pipeline/fake.py`, `tests/test_fake_pipeline.py`
- Test: `tests/test_fake_pipeline.py`

**Interfaces:**
- Consumes: `decode_and_resize`, `FaceDetectionError`
- Produces:
  - `FacePipeline` protocol: `embed_bgr(image_bgr: np.ndarray) -> np.ndarray`, `ready() -> bool`
  - `FakeFacePipeline`: deterministic L2-normalized 512-d vector from image mean pixels; raises `FaceDetectionError` if image is all zeros (simulate no face) OR if `force_faces` attribute set

Fake rules (document in code):
- If `self.faces_count == 0` or mean pixel value `< 1.0` → `FaceDetectionError("no face detected")`
- If `self.faces_count > 1` → `FaceDetectionError("multiple faces detected")`
- Else deterministic L2-normalized 512-d vector from image content
- Default `faces_count = 1`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_fake_pipeline.py
import numpy as np
import pytest
from app.errors import FaceDetectionError
from app.pipeline.fake import FakeFacePipeline

def test_embed_deterministic_and_unit_norm():
    pipe = FakeFacePipeline()
    img = np.full((64, 64, 3), 120, dtype=np.uint8)
    a = pipe.embed_bgr(img)
    b = pipe.embed_bgr(img)
    assert a.shape == (512,)
    assert a.dtype == np.float32
    assert np.allclose(a, b)
    assert np.linalg.norm(a) == pytest.approx(1.0, abs=1e-5)

def test_blank_image_no_face():
    pipe = FakeFacePipeline()
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    with pytest.raises(FaceDetectionError):
        pipe.embed_bgr(img)
```

- [ ] **Step 2: Run — expect fail**

Run: `pytest tests/test_fake_pipeline.py -v`  
Expected: FAIL

- [ ] **Step 3: Implement**

```python
# app/pipeline/base.py
from typing import Protocol
import numpy as np

class FacePipeline(Protocol):
    def embed_bgr(self, image_bgr: np.ndarray) -> np.ndarray: ...
    def ready(self) -> bool: ...
```

```python
# app/pipeline/fake.py
import numpy as np
from app.errors import FaceDetectionError

class FakeFacePipeline:
    def __init__(self, faces_count: int = 1) -> None:
        self.faces_count = faces_count

    def ready(self) -> bool:
        return True

    def embed_bgr(self, image_bgr: np.ndarray) -> np.ndarray:
        if self.faces_count == 0 or float(image_bgr.mean()) < 1.0:
            raise FaceDetectionError("no face detected")
        if self.faces_count > 1:
            raise FaceDetectionError("multiple faces detected")
        flat = image_bgr.reshape(-1).astype(np.float32)
        rng = np.random.default_rng(int(flat.sum()) % (2**32 - 1))
        v = rng.standard_normal(512, dtype=np.float32)
        v /= np.linalg.norm(v) + 1e-12
        return v
```

```python
# app/pipeline/__init__.py
from app.pipeline.fake import FakeFacePipeline

__all__ = ["FakeFacePipeline"]
```

- [ ] **Step 4: Run — expect pass**

Run: `pytest tests/test_fake_pipeline.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/pipeline tests/test_fake_pipeline.py
git commit -m "feat: add FacePipeline protocol and fake implementation"
```

---

### Task 6: Wire HTTP API (enroll/update/delete/verify/search) with fakes

**Files:**
- Modify: `app/main.py`, `app/api/routes.py`, `app/deps.py` (create), `tests/conftest.py`
- Create: `tests/test_api_faces.py`
- Test: `tests/test_api_faces.py`

**Interfaces:**
- Consumes: `Settings`, `FacePipeline`, `FaceStore`, `decode_and_resize`, matching helpers, domain errors
- Produces: routes as in spec; exception handlers mapping:
  - `InvalidImageError` → 400
  - `FaceNotFoundError` → 404
  - `FaceDetectionError` → 422
  - `DependencyUnavailableError` → 503

**Dependency injection:** store pipeline/store on `app.state` in `create_app(testing=True)` using `FakeFacePipeline` + `MemoryFaceStore`. Production path constructed in Task 7–8.

- [ ] **Step 1: Write failing API tests**

```python
# tests/test_api_faces.py
import numpy as np
import cv2
from fastapi.testclient import TestClient
from app.main import create_app
from app.pipeline.fake import FakeFacePipeline
from app.store.memory import MemoryFaceStore

def _client():
    app = create_app(testing=True)
    app.state.pipeline = FakeFacePipeline()
    app.state.store = MemoryFaceStore()
    return TestClient(app)

def _img_bytes(val: int = 120) -> bytes:
    img = np.full((80, 80, 3), val, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()

def test_enroll_update_verify_search_delete_flow():
    c = _client()
    r = c.post("/v1/faces", files={"image": ("a.png", _img_bytes(120), "image/png")})
    assert r.status_code == 201
    face_id = r.json()["face_id"]

    r2 = c.put(f"/v1/faces/{face_id}", files={"image": ("b.png", _img_bytes(121), "image/png")})
    assert r2.status_code == 200
    assert r2.json()["updated"] is True

    r3 = c.post(
        f"/v1/faces/{face_id}/verify",
        files={"image": ("c.png", _img_bytes(121), "image/png")},
    )
    assert r3.status_code == 200
    body = r3.json()
    assert body["face_id"] == face_id
    assert "score" in body and "match" in body

    # second enroll then search
    r4 = c.post("/v1/faces", files={"image": ("d.png", _img_bytes(121), "image/png")})
    assert r4.status_code == 201
    r5 = c.post(
        "/v1/faces/search",
        files={"image": ("e.png", _img_bytes(121), "image/png")},
        params={"limit": 5},
    )
    assert r5.status_code == 200
    assert "results" in r5.json()

    r6 = c.delete(f"/v1/faces/{face_id}")
    assert r6.status_code == 200
    assert r6.json()["deleted"] is True
    r7 = c.delete(f"/v1/faces/{face_id}")
    assert r7.status_code == 404

def test_verify_unknown_404():
    c = _client()
    r = c.post(
        "/v1/faces/00000000-0000-0000-0000-000000000000/verify",
        files={"image": ("a.png", _img_bytes(), "image/png")},
    )
    assert r.status_code == 404

def test_empty_image_400():
    c = _client()
    r = c.post("/v1/faces", files={"image": ("a.png", b"", "image/png")})
    assert r.status_code == 400

def test_no_face_422():
    c = _client()
    r = c.post("/v1/faces", files={"image": ("a.png", _img_bytes(0), "image/png")})
    assert r.status_code == 422
```

Also add test for search limit clamp and health `503` when store.ping is False — implement a tiny wrapper or monkeypatch in the same file:

```python
def test_health_503_when_store_down():
    app = create_app(testing=True)
    app.state.pipeline = FakeFacePipeline()
    store = MemoryFaceStore()
    store.ping = lambda: False  # type: ignore
    app.state.store = store
    c = TestClient(app)
    r = c.get("/health")
    assert r.status_code == 503

def test_corrupt_image_400():
    c = _client()
    r = c.post("/v1/faces", files={"image": ("a.png", b"not-an-image", "image/png")})
    assert r.status_code == 400

def test_multiple_faces_422():
    app = create_app(testing=True)
    app.state.pipeline = FakeFacePipeline(faces_count=2)
    app.state.store = MemoryFaceStore()
    c = TestClient(app)
    r = c.post("/v1/faces", files={"image": ("a.png", _img_bytes(120), "image/png")})
    assert r.status_code == 422

def test_search_limit_clamped_to_max():
    c = _client()
    c.post("/v1/faces", files={"image": ("a.png", _img_bytes(120), "image/png")})
    r = c.post(
        "/v1/faces/search",
        files={"image": ("b.png", _img_bytes(120), "image/png")},
        params={"limit": 999},
    )
    assert r.status_code == 200
    assert len(r.json()["results"]) <= 20

def test_verify_503_when_store_raises_dependency():
    from app.errors import DependencyUnavailableError
    app = create_app(testing=True)
    app.state.pipeline = FakeFacePipeline()
    store = MemoryFaceStore()
    def boom(*args, **kwargs):
        raise DependencyUnavailableError("qdrant down")
    store.get = boom  # type: ignore
    app.state.store = store
    c = TestClient(app)
    r = c.post(
        "/v1/faces/11111111-1111-1111-1111-111111111111/verify",
        files={"image": ("a.png", _img_bytes(), "image/png")},
    )
    assert r.status_code == 503
```

- [ ] **Step 2: Run — expect fail**

Run: `pytest tests/test_api_faces.py -v`  
Expected: FAIL (routes missing)

- [ ] **Step 3: Implement deps, handlers, routes**

```python
# app/deps.py
from fastapi import Request
from app.config import Settings, get_settings

def settings_dep() -> Settings:
    return get_settings()

def pipeline_dep(request: Request):
    return request.app.state.pipeline

def store_dep(request: Request):
    return request.app.state.store
```

Implement `app/api/routes.py` with:

- `GET /health` — if not `pipeline.ready()` or not `store.ping()` → raise `HTTPException(503)`
- `POST /v1/faces` — read upload, enforce `max_upload_bytes` via `decode_and_resize`, `embed_bgr`, `uuid4()`, `store.upsert`, return 201
- `PUT /v1/faces/{face_id}` — if `store.get` is None → `FaceNotFoundError`; else embed + upsert; 200
- `DELETE /v1/faces/{face_id}` — if not `store.delete` → 404; else 200
- `POST /v1/faces/{face_id}/verify` — get vector or 404; embed probe; cosine; return match
- `POST /v1/faces/search` — query params `limit` (default settings, clamp 1..max), `min_score` optional; embed; `store.search`; return results

Register exception handlers in `create_app`:

```python
from fastapi import Request
from fastapi.responses import JSONResponse
from app.errors import (
    InvalidImageError, FaceDetectionError, FaceNotFoundError, DependencyUnavailableError,
)

@app.exception_handler(InvalidImageError)
async def _(request: Request, exc: InvalidImageError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})

@app.exception_handler(FaceNotFoundError)
async def _(request: Request, exc: FaceNotFoundError):
    return JSONResponse(status_code=404, content={"detail": str(exc)})

@app.exception_handler(FaceDetectionError)
async def _(request: Request, exc: FaceDetectionError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})

@app.exception_handler(DependencyUnavailableError)
async def _(request: Request, exc: DependencyUnavailableError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})
```

Use `uuid.uuid4()` for new face ids. Read upload with:

```python
data = await image.read()
```

For sync store/pipeline inside async routes, call them directly in MVP (CPU work); acceptable for this plan.

Update `create_app` so `testing=True` installs fake pipeline + memory store by default if not already set.

Update health test if status becomes conditional — keep returning 200 under testing defaults.

- [ ] **Step 4: Run API + prior tests**

Run: `pytest tests/ -v`  
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add app/api app/deps.py app/main.py tests/test_api_faces.py tests/conftest.py tests/test_health.py
git commit -m "feat: add face enroll update delete verify search API"
```

---

### Task 7: QdrantFaceStore

**Files:**
- Create: `app/store/qdrant.py`, `tests/test_qdrant_store.py`
- Modify: `app/store/__init__.py`
- Test: `tests/test_qdrant_store.py` (unit tests with mocked `QdrantClient`)

**Interfaces:**
- Consumes: `Settings.qdrant_url`, `Settings.qdrant_collection`, `Settings.embedding_dim`
- Produces: `QdrantFaceStore` with same methods as `MemoryFaceStore`
- On init: `ensure_collection()` — create collection with vectors size 512, distance Cosine if missing
- Point id: UUID string accepted by Qdrant (`PointStruct(id=face_id, vector=..., payload={...})`)
- `search`: use client search; convert Qdrant score to cosine similarity consistent with client docs (cosine distance mode returns similarity scores in recent qdrant-client — assert in test via mock return values)
- Wrap connection errors as `DependencyUnavailableError` in `ping` / operations

- [ ] **Step 1: Write failing mock-based tests**

```python
# tests/test_qdrant_store.py
from unittest.mock import MagicMock
import numpy as np
import pytest
from app.store.qdrant import QdrantFaceStore

def test_upsert_calls_client():
    client = MagicMock()
    store = QdrantFaceStore(client=client, collection="faces", dim=512)
    v = np.ones(512, dtype=np.float32)
    store.upsert("11111111-1111-1111-1111-111111111111", v)
    assert client.upsert.called

def test_get_missing_returns_none():
    client = MagicMock()
    client.retrieve.return_value = []
    store = QdrantFaceStore(client=client, collection="faces", dim=512)
    assert store.get("11111111-1111-1111-1111-111111111111") is None

def test_delete_false_when_missing():
    client = MagicMock()
    client.retrieve.return_value = []
    store = QdrantFaceStore(client=client, collection="faces", dim=512)
    assert store.delete("11111111-1111-1111-1111-111111111111") is False
    client.delete.assert_not_called()
```

- [ ] **Step 2: Run — expect fail**

Run: `pytest tests/test_qdrant_store.py -v`  
Expected: FAIL

- [ ] **Step 3: Implement `QdrantFaceStore`**

Use `qdrant_client.QdrantClient`, `VectorParams(size=dim, distance=Distance.COSINE)`, `PointStruct`, `models.Filter` only if needed.  
`ping`: `client.get_collections()` or `client.health_check()` if available; return False on exception.

- [ ] **Step 4: Run — expect pass**

Run: `pytest tests/test_qdrant_store.py tests/test_memory_store.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/store/qdrant.py app/store/__init__.py tests/test_qdrant_store.py
git commit -m "feat: add Qdrant-backed face vector store"
```

---

### Task 8: Real ONNX/InsightFace pipeline + model download script

**Files:**
- Create: `app/pipeline/onnx_insightface.py`, `scripts/download_models.py`
- Modify: `app/pipeline/__init__.py`, `app/main.py` (production wiring)
- Test: keep using Fake in pytest; optional smoke marked `pytest.mark.slow` skipped by default

**Interfaces:**
- Produces: `InsightFacePipeline`:
  - `__init__(self, model_dir: str, model_name: str = "buffalo_sc")`
  - loads `insightface.app.FaceAnalysis` with `providers=["CPUExecutionProvider"]`
  - `embed_bgr`: `faces = app.get(image_bgr)`; if len==0 → `FaceDetectionError("no face detected")`; if len>1 → `FaceDetectionError("multiple faces detected")`; return `faces[0].normed_embedding.astype(np.float32)` (512-d)
  - `ready`: True after successful prepare

- [ ] **Step 1: Write skipped smoke test stub**

```python
# tests/test_insightface_smoke.py
import os
import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_INSIGHTFACE_SMOKE") != "1",
    reason="set RUN_INSIGHTFACE_SMOKE=1 with models downloaded",
)

def test_pipeline_ready():
    from app.pipeline.onnx_insightface import InsightFacePipeline
    pipe = InsightFacePipeline(model_dir="./models")
    assert pipe.ready()
```

- [ ] **Step 2: Implement pipeline + download script**

```python
# scripts/download_models.py
"""Download InsightFace buffalo_sc into MODEL_DIR (default ./models)."""
from pathlib import Path
from insightface.app import FaceAnalysis

def main() -> None:
    root = Path(os.environ.get("MODEL_DIR", "./models"))
    root.mkdir(parents=True, exist_ok=True)
    app = FaceAnalysis(name="buffalo_sc", root=str(root))
    app.prepare(ctx_id=-1, det_size=(640, 640))
    print(f"models ready under {root}")

if __name__ == "__main__":
    import os
    main()
```

Wire production in `create_app(testing=False)` lifespan:

1. Load settings  
2. Construct `InsightFacePipeline(settings.model_dir)` — on failure set pipeline stub that `ready()==False` or raise at startup (prefer fail startup with clear log)  
3. Construct `QdrantFaceStore` from `QdrantClient(url=settings.qdrant_url)` and `ensure_collection()`  
4. Attach to `app.state`

For `testing=True`, keep Fake + Memory.

- [ ] **Step 3: Manual sanity (not CI)**

```bash
python scripts/download_models.py
RUN_INSIGHTFACE_SMOKE=1 pytest tests/test_insightface_smoke.py -v
```

Expected: PASS if models/network OK

- [ ] **Step 4: Commit**

```bash
git add app/pipeline app/main.py scripts/download_models.py tests/test_insightface_smoke.py
git commit -m "feat: add InsightFace CPU pipeline and model download script"
```

---

### Task 9: Docker Compose + README + Swagger manual checklist

**Files:**
- Create: `docker-compose.yml`, `README.md`
- Modify: none required for tests

**Interfaces:**
- Produces: local run docs; `docker compose up -d qdrant`

- [ ] **Step 1: Add compose file**

```yaml
# docker-compose.yml
services:
  qdrant:
    image: qdrant/qdrant:v1.12.5
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage

volumes:
  qdrant_data:
```

- [ ] **Step 2: Write README**

Include:
- What iM.FaceId is (1 paragraph)
- `python -m venv .venv && pip install -r requirements.txt`
- `docker compose up -d qdrant`
- `python scripts/download_models.py`
- `uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`
- Open `http://127.0.0.1:8000/docs`
- Swagger checklist: enroll → update → search → verify → delete
- Env var table from spec
- Note: internal network only; no auth in MVP

- [ ] **Step 3: Run full unit/API suite**

Run: `pytest tests/ -v -k "not insightface_smoke"`  
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add docker-compose.yml README.md
git commit -m "docs: add Docker Compose Qdrant and local Swagger runbook"
```

---

## Spec coverage checklist

| Spec item | Task |
|-----------|------|
| Enroll UUID + Qdrant upsert | 6, 7, 8 |
| Update overwrite | 6, 7 |
| Delete + 404 if missing | 6, 7 |
| Verify 1:1 cosine | 2, 6 |
| Search top-K | 4, 6, 7 |
| Health model+Qdrant | 1, 6, 8 |
| Errors 400/404/422/503 | 3, 6 |
| SCRFD/buffalo_sc CPU | 8 |
| Swagger `/docs` | 1, 9 (FastAPI default) |
| Config env vars | 1, 9 |
| docker-compose Qdrant | 9 |
| No image disk persist | 6 (bytes → memory only) |
| Tests unit/API | 2–7 |

## Execution handoff

Plan complete after save + commit. Human chooses Subagent-driven vs Native before coding continues.
