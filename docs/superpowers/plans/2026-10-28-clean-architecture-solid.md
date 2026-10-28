# Clean Architecture + SOLID Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure `app/` into four layers (domain / application / infrastructure / api) with SOLID-compliant seams, deleting the 541-line `app/api/routes.py` monolith, while preserving every HTTP response byte-for-byte.

**Architecture:** Dependencies point inward only. `app/domain/` holds errors, value objects, `Protocol` ports and pure check strategies — it imports nothing from the other three layers. `app/application/` holds 9 use-case classes and 3 query services that depend only on domain ports. `app/infrastructure/` holds the concrete adapters (InsightFace, ONNX, Qdrant, SQLAlchemy, OpenCV, `threading.Lock`). `app/api/` holds 4 tagged routers, pydantic schemas, FastAPI dependency providers and exception handlers; routers contain no business logic — they build a params object, call one use case, and shape the response. `app/bootstrap/` wires everything.

**Tech Stack:** Python 3.11+, FastAPI, pydantic / pydantic-settings, SQLAlchemy 2.0, Qdrant, InsightFace, onnxruntime, OpenCV, numpy, pytest + `fastapi.testclient.TestClient`.

**Spec:** `docs/superpowers/specs/2026-10-28-clean-architecture-solid-design.md`

## Global Constraints

- **Behaviour-preserving refactor.** No endpoint path, HTTP status, response field name, or error `detail` string may change. Every error message string is copied verbatim from the current code, including f-string format specifiers (`:.1f`, `:.2f`).
- **Dependency rule.** `app/domain/` imports only stdlib, `numpy`, and other `app/domain/` modules — never `app.application`, `app.infrastructure`, `app.api`, `app.config`, or `fastapi`. `app/application/` imports `app.domain` and stdlib only — never `fastapi`, `sqlalchemy`, `cv2`, `qdrant_client`, or `app.config`. Violations are caught by the import-guard test in Task 11.
- **`Settings` stays at `app/config.py`.** It is a composition-root concern. The application layer never sees it; routers translate `Settings` + request body into frozen params dataclasses (`GateParams`, `MultiframeParams`).
- **No `HTTPException` outside `app/api/`.** Domain errors from `app/domain/errors.py` are raised inward and translated to statuses by `app/api/exception_handlers.py`.
- **Ports are `typing.Protocol`**, structurally satisfied. Adapters gain no base classes and no new declarations.
- **Route registration order is load-bearing:** `GET /v1/faces/count` MUST be registered before `GET /v1/faces/{face_id}`, else `count` is swallowed as a `face_id`.
- **Swagger tags:** exactly four sections — `Faces` (10 routes), `Liveness` (1), `Groups` (4), `System` (2) — declared in `openapi_tags` in that order.
- **No opportunistic fixes.** The group-rebuild TOCTOU race stays open; only the `RebuildLock` port is extracted (adapter remains `threading.Lock`). Do not add a DB constraint or change `GroupRebuildService` semantics.
- **Commit after every task.** Never commit `.env`. `.claude/` stays untracked.
- **graphify:** this plan's executor must run `graphify query "<question>"` before grepping or reading source, and `graphify update .` after each task's code change.

## Review Focus

Five behaviours the spec assumes but that no naive implementation of it exercises. Each line names the test added to the owning task.

1. **`liveness=false` must skip the entire check chain.** Today `_run_quality_and_liveness` returns `None` and runs zero quality checks when `body.liveness` is false — a blurry, dark, 10px face enrolls fine. An implementer who wires the chain unconditionally silently adds validation and breaks existing clients. → test in **Task 5** (`test_enroll_with_liveness_false_runs_no_checks`).
2. **"empty face crop" is raised unconditionally today**, before any threshold guard. With `blur_min_variance=0`, `brightness_min=0`, `brightness_max=255` a zero-area crop still raises `ImageQualityError("image quality failed: empty face crop")`. A lazily-computed crop or a conditionally-included check drops this. → test in **Task 4** (`test_face_crop_check_runs_with_all_thresholds_relaxed`).
3. **`planarity_residual` is a number, not `None`, when `min_planarity == 0` but `min_view_change > 0`.** The current early return needs *both* `<= 0`. → test in **Task 4** (`test_planarity_residual_reported_when_only_view_change_enabled`) and **Task 6** (`test_multiframe_response_reports_residual_with_planarity_disabled`).
4. **`count` before `{face_id}`.** A router whose decorators are reordered returns `404 face_id 'count' not found` instead of `200 {"count": N}`. → test in **Task 9** (`test_faces_count_not_shadowed_by_face_id_route`).
5. **Unavailable liveness checker must yield `503` with detail exactly `"liveness model not available (run scripts/download_liveness_model.py)"`,** not a leaked `RuntimeError` → 500. `UnavailableLivenessChecker.ready()` is `False` and both its other methods raise `RuntimeError`, so the `ready()` probe must run *before* any scoring call — and only when `liveness=True`. → test in **Task 5** (`test_unavailable_liveness_checker_raises_dependency_unavailable`).

---

## File Structure

```
app/
  domain/
    errors.py                 # moved from app/errors.py (unchanged content)
    face.py                   # DetectedFace (moved from pipeline/base.py)
    matching/                 # cosine.py, grouping.py, ann_grouping.py
                               # (package move: git mv app/matching app/domain/matching)
    planarity.py              # moved from app/pipeline/planarity.py
    ports/
      __init__.py
      store.py                # FaceReader, FaceWriter, FaceSearcher, FaceCatalog, HealthProbe
      pipeline.py             # FaceDetector, FaceEmbedder, PipelineProbe
      liveness.py             # LivenessScorer
      groups.py               # GroupsReader, GroupsWriter
      imaging.py              # ImageDecoder
      identity.py             # IdentityFactory
      rebuild_lock.py         # RebuildLock
      unit_of_work.py         # UnitOfWork
    checks/
      __init__.py
      base.py                 # Frame, FrameSequence, FrameCheck, SequenceCheck
      quality.py              # MinSize, FaceCrop, Blur, BrightnessMin, BrightnessMax
      liveness.py             # LivenessScoreCheck, MeanLivenessCheck, LivenessReadyCheck
      motion.py               # InterFrameDistance, KeypointsAvailable, ViewChange, Planarity
      chains.py               # build_quality_chain, build_single_frame_chain,
                               # planarity_enabled, build_sequence_chain
  application/
    params.py                 # GateParams, MultiframeParams
    gate.py                   # FaceGate
    use_cases/                # 9 classes
    queries/                  # 3 classes
  infrastructure/
    imaging.py                # moved from app/imaging.py
    identity.py               # Uuid4IdentityFactory
    locks/thread_lock.py      # ThreadRebuildLock
    pipeline/                 # onnx_insightface.py, fake.py, onnx_liveness.py,
                              # fake_liveness.py, metrics.py, image_quality.py (shim)
    store/                    # qdrant.py, memory.py
    db/                       # models.py, session.py, repository.py, unit_of_work.py
    jobs/group_rebuild.py     # orchestration adapter
  api/
    routers/faces.py liveness.py groups.py system.py
    schemas/faces.py liveness.py groups.py system.py
    dependencies.py
    exception_handlers.py
  bootstrap/
    lifespan.py
    app_factory.py
  config.py
  main.py                     # `from app.bootstrap.app_factory import create_app`
```

---

### Task 0: Prerequisites — merge GPU branch, pin the contract

No production code changes. This task creates the safety net every later task is graded against.

**Files:**
- Modify: `app/main.py`, `app/pipeline/*` (via merge only)
- Create: `tests/test_openapi_contract.py`
- Create: `tests/contract/openapi_baseline.json`

**Interfaces:**
- Consumes: nothing.
- Produces: `tests/contract/openapi_baseline.json` — the frozen OpenAPI document every later task must keep matching except for added `tags`.

- [ ] **Step 1: Merge the GPU branch into main, then rebase this branch**

```bash
git fetch --all
git checkout main
git merge --no-ff feat/face-device-auto-gpu -m "Merge branch 'feat/face-device-auto-gpu'"
python -m pytest -q
git checkout refactor/clean-architecture-solid
git rebase main
```

Expected: tests pass on main after the merge; the rebase replays the single spec commit cleanly.

- [ ] **Step 2: Write the OpenAPI contract test**

Create `tests/test_openapi_contract.py`:

```python
import json
from pathlib import Path

BASELINE = Path(__file__).parent / "contract" / "openapi_baseline.json"


def _signature(spec: dict) -> dict:
    """Path + method + status codes + response schema refs, ignoring tags/summaries."""
    out: dict = {}
    for path, methods in spec["paths"].items():
        for method, op in methods.items():
            key = f"{method.upper()} {path}"
            out[key] = {
                "statuses": sorted(op.get("responses", {}).keys()),
                "request": json.dumps(
                    op.get("requestBody", {})
                    .get("content", {})
                    .get("application/json", {})
                    .get("schema", {}),
                    sort_keys=True,
                ),
                "response": json.dumps(
                    op["responses"]
                    .get("200", {})
                    .get("content", {})
                    .get("application/json", {})
                    .get("schema", {}),
                    sort_keys=True,
                ),
            }
    return out


def test_openapi_signature_matches_baseline(client):
    current = _signature(client.get("/openapi.json").json())
    expected = _signature(json.loads(BASELINE.read_text()))
    assert current == expected


def test_component_schemas_unchanged(client):
    current = client.get("/openapi.json").json()["components"]["schemas"]
    expected = json.loads(BASELINE.read_text())["components"]["schemas"]
    assert sorted(current) == sorted(expected)
    for name in expected:
        assert current[name] == expected[name], f"schema {name} drifted"
```

- [ ] **Step 3: Generate the baseline from the current code**

```bash
mkdir -p tests/contract
python - <<'PY'
import json, pathlib
from app.main import create_app
spec = create_app(testing=True).openapi()
pathlib.Path("tests/contract/openapi_baseline.json").write_text(json.dumps(spec, indent=2, sort_keys=True))
print(len(spec["paths"]), "paths")
PY
```

Expected: prints `13 paths` — `len(spec["paths"])` counts path objects, not operations. The 17 `@router.*` operations are spread across 13 distinct path objects (several paths serve both GET and POST/PUT/DELETE).

- [ ] **Step 4: Run the contract test — it must pass against unmodified code**

Run: `python -m pytest tests/test_openapi_contract.py -v`
Expected: 2 PASSED. (A failure here means the baseline generation and the fixture disagree — fix before proceeding.)

- [ ] **Step 5: Record the full test baseline**

```bash
python -m pytest -q | tail -5
```

Write the pass count into the commit message. Every later task must match or exceed it.

- [ ] **Step 6: Commit**

```bash
git add tests/test_openapi_contract.py tests/contract/openapi_baseline.json
git commit -m "test: freeze OpenAPI contract before clean-architecture refactor

Baseline: <N> passed. Later refactor tasks must keep this green."
```

Substitute the real number for `<N>` from the Step 5 output before committing.

---

### Task 1: `app/domain/errors.py` + `app/domain/face.py`

The innermost leaves. Nothing depends on anything yet, so this is a pure move with re-export shims so the rest of the tree keeps importing.

**Files:**
- Create: `app/domain/__init__.py`, `app/domain/errors.py`, `app/domain/face.py`
- Modify: `app/errors.py` (becomes a re-export shim), `app/pipeline/base.py` (re-exports `DetectedFace`)
- Test: `tests/test_domain_layering.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `app.domain.errors`: `InvalidImageError`, `FaceDetectionError(message, *, code="face_detection")`, `FaceNotFoundError`, `LivenessFailedError`, `ImageQualityError`, `DependencyUnavailableError` — all `Exception` subclasses, `str(exc)` is the message.
  - `app.domain.face`: `DetectedFace` frozen dataclass with `embedding: np.ndarray`, `bbox: np.ndarray`, `kps: np.ndarray | None = None`.

- [ ] **Step 1: Write the layering test**

Create `tests/test_domain_layering.py`:

```python
import ast
import pathlib

DOMAIN = pathlib.Path("app/domain")
FORBIDDEN_PREFIXES = ("app.application", "app.infrastructure", "app.api", "app.config",
                      "app.main", "app.bootstrap")
FORBIDDEN_TOP = ("fastapi", "pydantic", "sqlalchemy", "cv2", "qdrant_client", "insightface",
                 "onnxruntime", "pydantic_settings")


def _imported_modules(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
    return names


def test_domain_package_exists():
    assert (DOMAIN / "errors.py").is_file()


def test_domain_imports_nothing_outward():
    offenders = []
    for path in DOMAIN.rglob("*.py"):
        for module in _imported_modules(path):
            if module.startswith(FORBIDDEN_PREFIXES) or module.split(".")[0] in FORBIDDEN_TOP:
                offenders.append(f"{path}: {module}")
    assert offenders == [], "domain layer reaches outward:\n" + "\n".join(offenders)
```

Without `test_domain_package_exists`, `test_domain_imports_nothing_outward` would pass vacuously on an unmodified tree: `app/domain` does not exist yet, so `DOMAIN.rglob` yields nothing and the `offenders == []` assertion holds for the wrong reason. The guard test makes the failure explicit.

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_domain_layering.py -v`
Expected: `test_domain_package_exists` FAILS (`app/domain` does not exist yet). `test_domain_imports_nothing_outward` passes vacuously for now — that is expected and gets exercised for real once Step 3 creates the package.

- [ ] **Step 3: Create the domain package and move errors**

Create `app/domain/__init__.py` (empty file).

Move `app/errors.py` to `app/domain/errors.py` with `git mv`, not a retype:

```bash
git mv app/errors.py app/domain/errors.py
```

The file's content is not retyped — the error `detail` strings and docstrings stay byte-identical, and git keeps the blame history. The next step recreates `app/errors.py` as a re-export shim, so the old import path keeps resolving.

- [ ] **Step 4: Turn `app/errors.py` into a shim**

`app/errors.py` no longer exists after the `git mv` above; recreate it as a thin re-export shim so existing importers keep working. Replace what is now a nonexistent file with a new one:

```python
"""Deprecated import path. Kept so existing tests and adapters keep working.

New code imports from `app.domain.errors`.
"""

from app.domain.errors import (
    DependencyUnavailableError,
    FaceDetectionError,
    FaceNotFoundError,
    ImageQualityError,
    InvalidImageError,
    LivenessFailedError,
)

__all__ = [
    "DependencyUnavailableError",
    "FaceDetectionError",
    "FaceNotFoundError",
    "ImageQualityError",
    "InvalidImageError",
    "LivenessFailedError",
]
```

- [ ] **Step 5: Create `app/domain/face.py`**

```python
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class DetectedFace:
    embedding: np.ndarray
    bbox: np.ndarray  # [x1, y1, x2, y2]
    # (5, 2) eyes/nose/mouth-corner landmarks; None when the model pack has no
    # keypoint head. Only the planarity liveness check needs them.
    kps: np.ndarray | None = None
```

Drop the unused `Protocol` import — `FacePipeline` does not move here (it becomes split ports in Task 2).

- [ ] **Step 6: Make `app/pipeline/base.py` re-export**

```python
from app.domain.face import DetectedFace
from app.domain.ports.pipeline import FacePipeline

__all__ = ["DetectedFace", "FacePipeline"]
```

**This step depends on Task 2.** Until Task 2 lands, write only:

```python
from app.domain.face import DetectedFace
from typing import Protocol

import numpy as np


class FacePipeline(Protocol):
    def detect_bgr(self, image_bgr: np.ndarray) -> DetectedFace: ...
    def embed_bgr(self, image_bgr: np.ndarray) -> np.ndarray: ...
    def ready(self) -> bool: ...


__all__ = ["DetectedFace", "FacePipeline"]
```

- [ ] **Step 7: Run the full suite**

Run: `python -m pytest -q`
Expected: same pass count as Task 0 Step 5, plus the 2 new layering tests.

- [ ] **Step 8: Commit**

```bash
git add app/domain tests/test_domain_layering.py app/errors.py app/pipeline/base.py
git commit -m "refactor: move errors and DetectedFace into app/domain

Adds an AST-based layering test that fails if app/domain imports outward."
```

---

### Task 2: `app/domain/ports/` — the ISP split

Nine fat methods on `FaceStore` become five role interfaces. Because they are `Protocol`s, `QdrantFaceStore` and `MemoryFaceStore` satisfy all five without edits.

**Files:**
- Create: `app/domain/ports/__init__.py`, `store.py`, `pipeline.py`, `liveness.py`, `groups.py`, `imaging.py`, `identity.py`, `rebuild_lock.py`, `unit_of_work.py`
- Modify: `app/store/base.py` (re-export shim), `app/pipeline/base.py` (import `FacePipeline` from ports)
- Test: `tests/test_ports_satisfied.py`

**Interfaces:**
- Consumes: `app.domain.face.DetectedFace` (Task 1).
- Produces the full port surface every later task programs against:
  - `FaceReader.get(face_id: str) -> np.ndarray | None`
  - `FaceWriter.upsert(face_id: str, vector: np.ndarray) -> None`; `delete(face_id: str) -> bool`
  - `FaceSearcher.search(vector: np.ndarray, *, limit: int, min_score: float) -> list[tuple[str, float]]`
  - `FaceCatalog.count() -> int`; `list_ids(*, limit: int, offset: int) -> list[str]`; `list_all() -> list[tuple[str, np.ndarray]]`; `clear() -> int`
  - `HealthProbe.ping() -> bool`
  - `FaceDetector.detect_bgr(image_bgr) -> DetectedFace`; `FaceEmbedder.embed_bgr(image_bgr) -> np.ndarray`; `PipelineProbe.ready() -> bool`; `FacePipeline` = all three
  - `LivenessScorer.ready() -> bool`; `score_bgr(image_bgr, bbox=None) -> float` (**`bbox` is positional-or-keyword, matching `OnnxLivenessChecker`**)
  - `ImageDecoder.decode(data: bytes, *, max_side: int, max_bytes: int) -> np.ndarray`
  - `IdentityFactory.new_id() -> str`
  - `RebuildLock.acquire(blocking: bool = ...) -> bool`; `release() -> None`
  - `UnitOfWork.__enter__() -> GroupsReader & GroupsWriter`-ish session handle; see `unit_of_work.py` below
  - `GroupsReader` / `GroupsWriter` mirroring `GroupsRepository`

Verify every signature above against the real adapter before writing the file. Run:

```bash
graphify explain "FaceStore protocol methods"
graphify explain "GroupsRepository methods"
```

- [ ] **Step 1: Write the port-satisfaction test**

Create `tests/test_ports_satisfied.py`:

```python
from app.domain.ports.liveness import LivenessScorer
from app.domain.ports.pipeline import FaceDetector, FaceEmbedder, FacePipeline, PipelineProbe
from app.domain.ports.store import (
    FaceCatalog,
    FaceReader,
    FaceSearcher,
    FaceWriter,
    HealthProbe,
)
from app.pipeline.fake import FakeFacePipeline
from app.pipeline.fake_liveness import FakeLivenessChecker, UnavailableLivenessChecker
from app.store.memory import MemoryFaceStore


def test_fake_store_satisfies_every_store_role():
    store = MemoryFaceStore()
    for port in (FaceReader, FaceWriter, FaceSearcher, FaceCatalog, HealthProbe):
        assert isinstance(store, port), f"MemoryFaceStore does not satisfy {port.__name__}"


def test_fake_pipeline_satisfies_pipeline_roles():
    pipeline = FakeFacePipeline()
    for port in (FaceDetector, FaceEmbedder, PipelineProbe, FacePipeline):
        assert isinstance(pipeline, port), f"FakeFacePipeline does not satisfy {port.__name__}"


def test_both_liveness_checkers_satisfy_the_scorer_port():
    assert isinstance(FakeLivenessChecker(), LivenessScorer)
    assert isinstance(UnavailableLivenessChecker(), LivenessScorer)
```

Check `FakeFacePipeline`'s real constructor signature with `graphify explain "FakeFacePipeline"` — if it requires arguments, pass them.

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_ports_satisfied.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.domain.ports'`.

- [ ] **Step 3: Write `app/domain/ports/store.py`**

```python
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class FaceReader(Protocol):
    def get(self, face_id: str) -> np.ndarray | None: ...


@runtime_checkable
class FaceWriter(Protocol):
    def upsert(self, face_id: str, vector: np.ndarray) -> None: ...
    def delete(self, face_id: str) -> bool: ...


@runtime_checkable
class FaceSearcher(Protocol):
    def search(
        self, vector: np.ndarray, *, limit: int, min_score: float
    ) -> list[tuple[str, float]]: ...


@runtime_checkable
class FaceCatalog(Protocol):
    def count(self) -> int: ...
    def list_ids(self, *, limit: int, offset: int) -> list[str]: ...
    def list_all(self) -> list[tuple[str, np.ndarray]]: ...
    def clear(self) -> int: ...


@runtime_checkable
class HealthProbe(Protocol):
    def ping(self) -> bool: ...


@runtime_checkable
class FaceStore(FaceReader, FaceWriter, FaceSearcher, FaceCatalog, HealthProbe, Protocol):
    """The whole surface. Only the composition root and health endpoints need it."""
```

`@runtime_checkable` is required for the `isinstance` assertions in Step 1; note it only checks method *presence*, not signatures — that is exactly the guarantee this test is for.

- [ ] **Step 4: Write the remaining port modules**

`app/domain/ports/pipeline.py`:

```python
from typing import Protocol, runtime_checkable

import numpy as np

from app.domain.face import DetectedFace


@runtime_checkable
class FaceDetector(Protocol):
    def detect_bgr(self, image_bgr: np.ndarray) -> DetectedFace: ...


@runtime_checkable
class FaceEmbedder(Protocol):
    def embed_bgr(self, image_bgr: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class PipelineProbe(Protocol):
    def ready(self) -> bool: ...


@runtime_checkable
class FacePipeline(FaceDetector, FaceEmbedder, PipelineProbe, Protocol):
    """The whole pipeline surface, for the composition root and /stats."""
```

`app/domain/ports/liveness.py`:

```python
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class LivenessScorer(Protocol):
    def ready(self) -> bool: ...

    # bbox is positional-or-keyword to match OnnxLivenessChecker.score_bgr.
    def score_bgr(self, image_bgr: np.ndarray, bbox: np.ndarray | None = None) -> float: ...
```

`ensure_live` is deliberately **not** on the port: the domain check chain owns the threshold decision and raises `LivenessFailedError` itself. The adapters keep their `ensure_live` method; nothing in the new code calls it.

`app/domain/ports/imaging.py`:

```python
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ImageDecoder(Protocol):
    def decode(self, data: bytes, *, max_side: int, max_bytes: int) -> np.ndarray: ...
```

`app/domain/ports/identity.py`:

```python
from typing import Protocol, runtime_checkable


@runtime_checkable
class IdentityFactory(Protocol):
    def new_id(self) -> str: ...
```

`app/domain/ports/rebuild_lock.py`:

```python
from typing import Protocol, runtime_checkable


@runtime_checkable
class RebuildLock(Protocol):
    """Single-flight guard for group rebuilds.

    The shipped adapter is process-local (threading.Lock), so single-flight
    holds only within one worker process. See the spec's "Known limitation".
    `locked()` exists because `GroupRebuildService.execute()` calls it in its
    `finally` branch; a future distributed adapter will have to emulate it.
    """

    def acquire(self, blocking: bool = True) -> bool: ...
    def release(self) -> None: ...
    def locked(self) -> bool: ...
```

`app/domain/ports/groups.py` — mirror `GroupsRepository` exactly. Confirm the real signatures first, then write:

The row types are deliberately left unannotated. `get_run`, `list_runs`, `create_run` and
`latest_completed` all return SQLAlchemy ORM rows, and naming `GroupRun` here — even under
`if TYPE_CHECKING:` — makes `app/domain/` import `app.infrastructure`, which the layering guard
flags (`ast.walk` does not skip a `TYPE_CHECKING` block). Task 8's `RebuildRunView` and
`GroupSnapshotView` are where those rows become typed; this port stays structural.

```python
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class GroupsReader(Protocol):
    def get_run(self, run_id: str): ...
    def list_runs(self) -> list[Any]: ...
    def has_running(self) -> bool: ...
    def latest_completed(self): ...


@runtime_checkable
class GroupsWriter(Protocol):
    def create_run(self, *, threshold: float, top_k: int, min_size: int): ...
    def mark_failed(self, run_id: str, error: str) -> None: ...
    def save_completed(
        self, run_id: str, *, face_count: int, groups: list[list[str]], keep_runs: int = 2
    ) -> None: ...
```

`app/domain/ports/unit_of_work.py`:

```python
from contextlib import AbstractContextManager
from typing import Protocol, runtime_checkable

from app.domain.ports.groups import GroupsReader, GroupsWriter


@runtime_checkable
class GroupsRepositoryPort(GroupsReader, GroupsWriter, Protocol): ...


@runtime_checkable
class UnitOfWork(Protocol):
    """Opens a transaction and yields the groups repository bound to it.

    Commits on clean exit, rolls back on exception, always closes — the
    semantics of the existing app/db/session.py::session_scope, which the
    adapter in Task 3 delegates to rather than reimplementing.
    """

    def groups(self) -> AbstractContextManager[GroupsRepositoryPort]: ...
```

- [ ] **Step 5: Make `app/store/base.py` a shim**

```python
from app.domain.ports.store import (
    FaceCatalog,
    FaceReader,
    FaceSearcher,
    FaceStore,
    FaceWriter,
    HealthProbe,
)

__all__ = [
    "FaceCatalog",
    "FaceReader",
    "FaceSearcher",
    "FaceStore",
    "FaceWriter",
    "HealthProbe",
]
```

Then update `app/pipeline/base.py` to the final form from Task 1 Step 6 (importing `FacePipeline` from `app.domain.ports.pipeline`).

- [ ] **Step 6: Run the tests to verify they pass**

Run: `python -m pytest tests/test_ports_satisfied.py tests/test_domain_layering.py -v`
Expected: all PASS.

Then: `python -m pytest -q` — expected: baseline count + new tests.

- [ ] **Step 7: Commit**

```bash
git add app/domain/ports app/store/base.py app/pipeline/base.py tests/test_ports_satisfied.py
git commit -m "refactor: split FaceStore into five role interfaces (ISP)

Ports are typing.Protocol, so existing adapters satisfy them unchanged.
Old import paths re-export for compatibility."
```

---

### Task 3: Pure domain logic + `app/infrastructure/` adapters

Three mechanical relocations. `matching.py`, `grouping.py` and `planarity.py` are already pure — they move with `git mv` and no edit. Everything that touches a driver, a socket or a file handle moves under `infrastructure/`.

**Files:**
- Move: `app/matching/` (package: `cosine.py`, `grouping.py`, `ann_grouping.py`) → `app/domain/matching/`
- Move: `app/pipeline/planarity.py` → `app/domain/planarity.py`
- Move: `app/imaging.py` → `app/infrastructure/imaging.py`
- Move: `app/store/` → `app/infrastructure/store/`
- Move: `app/db/` → `app/infrastructure/db/`
- Move: `app/pipeline/{onnx_insightface,fake,onnx_liveness,fake_liveness,device,image_quality}.py` → `app/infrastructure/pipeline/`
- Create: `app/infrastructure/__init__.py`, `app/infrastructure/identity.py`, `app/infrastructure/locks/__init__.py`, `app/infrastructure/locks/thread_lock.py`, `app/infrastructure/db/unit_of_work.py`
- Create: shims `app/imaging.py`, `app/matching/cosine.py`, `app/matching/grouping.py`, `app/matching/ann_grouping.py`, `app/store/*.py`, `app/db/*.py`, `app/pipeline/*.py` re-exporting from the new homes (all deleted in Task 11)
- Test: `tests/test_infrastructure_adapters.py`

**Interfaces:**
- Consumes: all ports from Task 2.
- Produces:
  - `app.infrastructure.imaging.OpenCvImageDecoder` — `decode(data, *, max_side, max_bytes) -> np.ndarray`, delegating to the existing module-level `decode_and_resize`. Satisfies `ImageDecoder`.
  - `app.infrastructure.identity.Uuid4IdentityFactory` — `new_id() -> str` returning `str(uuid.uuid4())`. Satisfies `IdentityFactory`.
  - `app.infrastructure.locks.thread_lock.ThreadRebuildLock` — wraps one `threading.Lock`. Satisfies `RebuildLock`.
  - `app.infrastructure.db.unit_of_work.SqlAlchemyUnitOfWork(session_factory)` — `groups()` context manager yielding `GroupsRepository(session)`, built on `session_scope`.
  - `app.domain.planarity`: `planarity_residual(kps_a, kps_b)`, `view_change(kps)`, `view_change_spread(frames)`, `max_planarity_residual(frames)` — unchanged.

- [ ] **Step 1: Move the three pure modules**

```bash
mkdir -p app/infrastructure/locks app/infrastructure/pipeline app/infrastructure/store
touch app/infrastructure/__init__.py app/infrastructure/locks/__init__.py
git mv app/matching app/domain/matching
git mv app/pipeline/planarity.py app/domain/planarity.py
```

`app/matching/` is a package (`cosine.py`, `grouping.py`, `ann_grouping.py`), not a single module — move the directory. Do not edit the moved files' bodies beyond intra-package import prefixes (`app.matching.cosine` → `app.domain.matching.cosine`).

Note for later tasks: `cosine_distance`, `cosine_similarity` and `is_match` live in `app.domain.matching.cosine`, not in the package root. Import them from the submodule.

- [ ] **Step 2: Move the infrastructure modules**

```bash
git mv app/imaging.py app/infrastructure/imaging.py
git mv app/store/qdrant.py app/infrastructure/store/qdrant.py
git mv app/store/memory.py app/infrastructure/store/memory.py
git mv app/db/models.py app/infrastructure/db/models.py
git mv app/db/session.py app/infrastructure/db/session.py
git mv app/db/repository.py app/infrastructure/db/repository.py
git mv app/pipeline/onnx_insightface.py app/infrastructure/pipeline/onnx_insightface.py
git mv app/pipeline/fake.py app/infrastructure/pipeline/fake.py
git mv app/pipeline/onnx_liveness.py app/infrastructure/pipeline/onnx_liveness.py
git mv app/pipeline/fake_liveness.py app/infrastructure/pipeline/fake_liveness.py
git mv app/pipeline/image_quality.py app/infrastructure/pipeline/image_quality.py
git mv app/pipeline/device.py app/infrastructure/pipeline/device.py   # if the GPU merge added it
touch app/infrastructure/db/__init__.py app/infrastructure/store/__init__.py app/infrastructure/pipeline/__init__.py
```

Fix the resulting intra-package imports (`app.errors` → `app.domain.errors`, `app.pipeline.base` → `app.domain.face` / `app.domain.ports.pipeline`, `app.pipeline.planarity` → `app.domain.planarity`, `app.db.models` → `app.infrastructure.db.models`). Run `python -c "import app.main"` after each batch; it will name the first broken import.

- [ ] **Step 3: Write compatibility shims so existing tests keep importing**

`app/imaging.py`:

```python
from app.infrastructure.imaging import decode_and_resize

__all__ = ["decode_and_resize"]
```

`app/matching/*.py`: explicit re-exports from `app.domain.matching.*` — list the real public names, never `*`. `app/store/__init__.py`, `app/store/qdrant.py`, `app/store/memory.py`, `app/db/__init__.py`, `app/db/session.py`, `app/db/repository.py`, `app/db/models.py`, `app/pipeline/*.py`: the same pattern, one re-export line per public name.

These shims exist only so Tasks 3–10 can land one at a time without breaking the suite. Task 11 deletes every shim and rewrites the importers.

- [ ] **Step 4: Write the adapter test**

Create `tests/test_infrastructure_adapters.py`:

```python
import numpy as np
import pytest

from app.domain.errors import InvalidImageError
from app.domain.ports.identity import IdentityFactory
from app.domain.ports.imaging import ImageDecoder
from app.domain.ports.rebuild_lock import RebuildLock
from app.infrastructure.identity import Uuid4IdentityFactory
from app.infrastructure.imaging import OpenCvImageDecoder
from app.infrastructure.locks.thread_lock import ThreadRebuildLock


def test_decoder_satisfies_port_and_rejects_empty_bytes():
    decoder = OpenCvImageDecoder()
    assert isinstance(decoder, ImageDecoder)
    with pytest.raises(InvalidImageError):
        decoder.decode(b"", max_side=640, max_bytes=5_000_000)


def test_identity_factory_returns_distinct_uuid_strings():
    factory = Uuid4IdentityFactory()
    assert isinstance(factory, IdentityFactory)
    ids = {factory.new_id() for _ in range(50)}
    assert len(ids) == 50
    assert all(len(i) == 36 for i in ids)


def test_thread_lock_is_single_flight():
    lock = ThreadRebuildLock()
    assert isinstance(lock, RebuildLock)
    assert lock.acquire(blocking=False) is True
    assert lock.acquire(blocking=False) is False
    assert lock.locked() is True
    lock.release()
    assert lock.locked() is False
    assert lock.acquire(blocking=False) is True
    lock.release()
```

- [ ] **Step 5: Run it to verify it fails**

Run: `python -m pytest tests/test_infrastructure_adapters.py -v`
Expected: FAIL — `ImportError: cannot import name 'OpenCvImageDecoder'`.

- [ ] **Step 6: Write the three small adapters**

Append to `app/infrastructure/imaging.py` (keep `decode_and_resize` exactly as it is):

```python
class OpenCvImageDecoder:
    """ImageDecoder adapter over the module-level decode_and_resize."""

    def decode(self, data: bytes, *, max_side: int, max_bytes: int) -> np.ndarray:
        return decode_and_resize(data, max_side=max_side, max_bytes=max_bytes)
```

`app/infrastructure/identity.py`:

```python
import uuid


class Uuid4IdentityFactory:
    """IdentityFactory adapter. Replaces the inline uuid.uuid4() in the old routes."""

    def new_id(self) -> str:
        return str(uuid.uuid4())
```

`app/infrastructure/locks/thread_lock.py`:

```python
import threading


class ThreadRebuildLock:
    """Process-local RebuildLock.

    Single-flight holds within one worker process only. Running uvicorn with
    --workers > 1 lets one rebuild start per worker; see the spec's "Known
    limitation". Swapping in a distributed adapter is a config change here.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def acquire(self, blocking: bool = True) -> bool:
        return self._lock.acquire(blocking)

    def release(self) -> None:
        self._lock.release()

    def locked(self) -> bool:
        return self._lock.locked()
```

- [ ] **Step 7: Write the unit-of-work adapter**

`app/infrastructure/db/unit_of_work.py`:

```python
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy.orm import Session, sessionmaker

from app.infrastructure.db.repository import GroupsRepository
from app.infrastructure.db.session import session_scope


class SqlAlchemyUnitOfWork:
    """UnitOfWork adapter built on the existing session_scope contextmanager.

    session_scope already commits on clean exit, rolls back on exception and
    always closes; this class only binds a repository to that session.
    """

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    @contextmanager
    def groups(self) -> Iterator[GroupsRepository]:
        with session_scope(self._session_factory) as session:
            yield GroupsRepository(session)
```

Confirm `GroupsRepository.__init__` really takes a bare session (`graphify explain "GroupsRepository"`); if it takes a session factory instead, adapt the call, not the port.

- [ ] **Step 8: Run the tests to verify they pass**

Run: `python -m pytest tests/test_infrastructure_adapters.py -v`
Expected: 3 PASSED.

Run: `python -m pytest -q`
Expected: baseline + new tests, zero failures. The OpenAPI contract test must still be green — this task touched no routes.

- [ ] **Step 9: Commit**

```bash
graphify update .
git add -A app tests/test_infrastructure_adapters.py
git commit -m "refactor: relocate pure logic to domain, drivers to infrastructure

Adds ImageDecoder, IdentityFactory, RebuildLock and UnitOfWork adapters.
Old import paths re-export; the shims are removed in the final task."
```

---

### Task 4: `app/domain/checks/` — the OCP check chains

The heart of the refactor. `ensure_face_quality` (a 5-branch function) and `_run_multiframe_liveness`'s three gates become 11 one-rule classes behind two protocols. Adding a gate later means writing a class and adding one line to a chain builder — no existing file is edited.

Two protocols, not one: `MinSizeCheck` judges a single frame, `MeanLivenessCheck` judges a whole sequence. One protocol covering both would force half the implementations to ignore their argument — an LSP violation written into the interface.

`cv2` may not be imported from `app/domain/`, so the four OpenCV measurements sit behind a new `FaceImageMetrics` port whose adapter lives in infrastructure. The numbers are byte-identical because the adapter calls the same `cv2` functions in the same order.

**Files:**
- Create: `app/domain/ports/metrics.py`
- Create: `app/domain/checks/__init__.py`, `base.py`, `quality.py`, `liveness.py`, `motion.py`, `chains.py`
- Create: `app/infrastructure/pipeline/metrics.py`
- Create: `app/application/params.py` (Step 10 creates `GateParams`/`MultiframeParams` here)
- Modify: `app/infrastructure/pipeline/image_quality.py` (becomes a shim delegating to the chain)
- Test: `tests/test_domain_checks.py`

**Interfaces:**
- Consumes: `DetectedFace` (Task 1); `LivenessScorer` (Task 2); `app.domain.matching.cosine.cosine_distance`; `app.domain.planarity.view_change_spread`, `max_planarity_residual`.
- Produces:
  - `app.domain.ports.metrics.FaceImageMetrics` — `crop(image_bgr, bbox) -> np.ndarray`, `gray(crop_bgr) -> np.ndarray`, `laplacian_variance(gray) -> float`, `mean_brightness(gray) -> float`
  - `app.domain.checks.base.Frame(image_bgr, detected, metrics, scorer=None)` with `cached_property` `crop`, `gray`, `liveness_score`
  - `app.domain.checks.base.FrameSequence(frames: list[Frame])` with `cached_property` `liveness_scores`, `mean_liveness`, `max_pairwise_distance`, `keypoints`, `view_change_spread`, `planarity_residual`
  - `app.domain.checks.base.FrameCheck` / `SequenceCheck` protocols, each with `name: str` and `verify(...) -> None`
  - `app.domain.checks.chains.build_quality_chain(gate) -> list[FrameCheck]`, `build_single_frame_chain(gate) -> list[FrameCheck]`, `build_sequence_chain(params) -> list[SequenceCheck]`, `planarity_enabled(params) -> bool`
  - `app.infrastructure.pipeline.metrics.OpenCvFaceImageMetrics`
  - `app.application.params.GateParams`, `MultiframeParams`

- [ ] **Step 1: Write the failing check tests**

Create `tests/test_domain_checks.py`:

```python
import numpy as np
import pytest

from app.application.params import GateParams, MultiframeParams
from app.domain.checks.base import Frame, FrameSequence
from app.domain.checks.chains import (
    build_quality_chain,
    build_single_frame_chain,
    build_sequence_chain,
    planarity_enabled,
)
from app.domain.errors import DependencyUnavailableError, ImageQualityError, LivenessFailedError
from app.domain.face import DetectedFace
from app.infrastructure.pipeline.metrics import OpenCvFaceImageMetrics

METRICS = OpenCvFaceImageMetrics()


def _gate(**over) -> GateParams:
    base = dict(
        blur_min_variance=0.0,
        brightness_min=0.0,
        brightness_max=255.0,
        min_side_px=0.0,
        liveness_threshold=0.5,
    )
    base.update(over)
    return GateParams(**base)


def _frame(image_bgr: np.ndarray, bbox, kps=None, scorer=None) -> Frame:
    detected = DetectedFace(
        embedding=np.ones(512, dtype=np.float32),
        bbox=np.asarray(bbox, dtype=np.float32),
        kps=kps,
    )
    return Frame(image_bgr=image_bgr, detected=detected, metrics=METRICS, scorer=scorer)


def _run(chain, frame):
    for check in chain:
        check.verify(frame)


# --- Review Focus #2 -------------------------------------------------------
def test_face_crop_check_runs_with_all_thresholds_relaxed():
    """Empty crop is rejected even when every threshold is relaxed to off."""
    image = np.full((100, 100, 3), 128, dtype=np.uint8)
    frame = _frame(image, [50.0, 50.0, 50.0, 50.0])  # zero-area box -> empty crop
    chain = build_quality_chain(_gate())
    with pytest.raises(ImageQualityError, match=r"^image quality failed: empty face crop$"):
        _run(chain, frame)


def test_min_size_check_message_is_verbatim():
    image = np.full((100, 100, 3), 128, dtype=np.uint8)
    frame = _frame(image, [0.0, 0.0, 20.0, 20.0])
    chain = build_quality_chain(_gate(min_side_px=80.0))
    with pytest.raises(ImageQualityError) as exc:
        _run(chain, frame)
    assert str(exc.value) == (
        "image quality failed: face too small (min side=20.0px < 80.0px)"
    )


def test_min_size_check_absent_when_threshold_is_zero():
    names = [c.name for c in build_quality_chain(_gate(min_side_px=0.0))]
    assert "min_side" not in names
    assert "face_crop" in names


def test_blur_check_rejects_flat_crop():
    image = np.full((100, 100, 3), 128, dtype=np.uint8)  # laplacian variance == 0
    frame = _frame(image, [10.0, 10.0, 90.0, 90.0])
    chain = build_quality_chain(_gate(blur_min_variance=50.0))
    with pytest.raises(ImageQualityError) as exc:
        _run(chain, frame)
    assert str(exc.value) == (
        "image quality failed: blur too high (laplacian_var=0.00 < 50.00)"
    )


def test_brightness_min_check_message_is_verbatim():
    image = np.full((100, 100, 3), 10, dtype=np.uint8)
    frame = _frame(image, [10.0, 10.0, 90.0, 90.0])
    chain = build_quality_chain(_gate(brightness_min=40.0))
    with pytest.raises(ImageQualityError) as exc:
        _run(chain, frame)
    assert str(exc.value) == (
        "image quality failed: brightness too low (mean=10.0 < 40.0)"
    )


def test_brightness_max_check_message_is_verbatim():
    image = np.full((100, 100, 3), 240, dtype=np.uint8)
    frame = _frame(image, [10.0, 10.0, 90.0, 90.0])
    chain = build_quality_chain(_gate(brightness_max=220.0))
    with pytest.raises(ImageQualityError) as exc:
        _run(chain, frame)
    assert str(exc.value) == (
        "image quality failed: brightness too high (mean=240.0 > 220.0)"
    )


def test_quality_chain_order_matches_legacy_gate():
    chain = build_quality_chain(
        _gate(min_side_px=80.0, blur_min_variance=50.0,
              brightness_min=40.0, brightness_max=220.0)
    )
    assert [c.name for c in chain] == [
        "min_side", "face_crop", "blur", "brightness_min", "brightness_max",
    ]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/test_domain_checks.py -v`
Expected: all FAIL at collection with `ModuleNotFoundError: No module named 'app.domain.checks'`.

- [ ] **Step 3: Write the metrics port and its OpenCV adapter**

`app/domain/ports/metrics.py`:

```python
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class FaceImageMetrics(Protocol):
    """The four OpenCV measurements the quality checks need.

    Behind a port so app/domain/ never imports cv2. The adapter calls the
    same cv2 functions the legacy ensure_face_quality did, so the numbers
    (and therefore the error strings) are unchanged.
    """

    def crop(self, image_bgr: np.ndarray, bbox: np.ndarray) -> np.ndarray: ...
    def gray(self, crop_bgr: np.ndarray) -> np.ndarray: ...
    def laplacian_variance(self, gray: np.ndarray) -> float: ...
    def mean_brightness(self, gray: np.ndarray) -> float: ...
```

`app/infrastructure/pipeline/metrics.py`:

```python
import cv2
import numpy as np


def _crop_face_bgr(image_bgr: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Tight crop from InsightFace-style ``[x1, y1, x2, y2]`` bbox."""
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in np.asarray(bbox, dtype=np.float64).reshape(-1)[:4])
    ix1 = max(0, min(w - 1, int(np.floor(x1))))
    iy1 = max(0, min(h - 1, int(np.floor(y1))))
    ix2 = max(ix1 + 1, min(w, int(np.ceil(x2))))
    iy2 = max(iy1 + 1, min(h, int(np.ceil(y2))))
    return image_bgr[iy1:iy2, ix1:ix2]


class OpenCvFaceImageMetrics:
    """FaceImageMetrics adapter. Thin wrappers, no logic of their own."""

    def crop(self, image_bgr: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        return _crop_face_bgr(image_bgr, bbox)

    def gray(self, crop_bgr: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    def laplacian_variance(self, gray: np.ndarray) -> float:
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def mean_brightness(self, gray: np.ndarray) -> float:
        return float(gray.mean())
```

`_crop_face_bgr` now lives in `metrics.py`, not `image_quality.py` — the cropping arithmetic is defined exactly once.

- [ ] **Step 4: Make `image_quality.py` re-export `_crop_face_bgr`**

Add this line near the top of `app/infrastructure/pipeline/image_quality.py` (the shim created in Task 3):

```python
from app.infrastructure.pipeline.metrics import _crop_face_bgr
```

This is the import direction reversed from earlier drafts of this plan: `metrics.py` now owns the function, and `image_quality.py` re-exports it. This is what lets Task 11 delete `image_quality.py` without breaking `OpenCvFaceImageMetrics`, since `metrics.py` no longer depends on the module Task 11 removes.

- [ ] **Step 5: Write `app/domain/checks/base.py`**

```python
from dataclasses import dataclass, field
from functools import cached_property
from typing import Protocol, runtime_checkable

import numpy as np

from app.domain.face import DetectedFace
from app.domain.matching.cosine import cosine_distance
from app.domain.planarity import max_planarity_residual, view_change_spread
from app.domain.ports.liveness import LivenessScorer
from app.domain.ports.metrics import FaceImageMetrics


# Not frozen: cached_property needs a __dict__ to memoise into. Treat every
# field as read-only by convention — nothing in the codebase reassigns them.
@dataclass
class Frame:
    """One decoded image plus its detection, with measurements computed once."""

    image_bgr: np.ndarray
    detected: DetectedFace
    metrics: FaceImageMetrics
    scorer: LivenessScorer | None = None

    @cached_property
    def min_side(self) -> float:
        x1, y1, x2, y2 = (float(v) for v in self.detected.bbox[:4])
        return min(max(0.0, x2 - x1), max(0.0, y2 - y1))

    @cached_property
    def crop(self) -> np.ndarray:
        return self.metrics.crop(self.image_bgr, self.detected.bbox)

    @cached_property
    def gray(self) -> np.ndarray:
        return self.metrics.gray(self.crop)

    @cached_property
    def liveness_score(self) -> float:
        if self.scorer is None:  # pragma: no cover - wiring error, not an input
            raise RuntimeError("Frame.liveness_score requires a scorer")
        return self.scorer.score_bgr(self.image_bgr, self.detected.bbox)


@dataclass
class FrameSequence:
    """Several frames judged together. Metrics are computed once and shared."""

    frames: list[Frame]

    @cached_property
    def liveness_scores(self) -> list[float]:
        return [f.liveness_score for f in self.frames]

    @cached_property
    def mean_liveness(self) -> float:
        scores = self.liveness_scores
        return float(sum(scores) / len(scores))

    @cached_property
    def max_pairwise_distance(self) -> float:
        embeddings = [f.detected.embedding for f in self.frames]
        return max(
            cosine_distance(embeddings[i], embeddings[j])
            for i in range(len(embeddings))
            for j in range(i + 1, len(embeddings))
        )

    @cached_property
    def keypoints(self) -> list[np.ndarray | None]:
        return [f.detected.kps for f in self.frames]

    @cached_property
    def view_change_spread(self) -> float:
        return view_change_spread(self.keypoints)

    @cached_property
    def planarity_residual(self) -> float:
        return max_planarity_residual(self.keypoints)


@runtime_checkable
class FrameCheck(Protocol):
    """One rule about one frame. Raises a domain error, or returns None."""

    name: str

    def verify(self, frame: Frame) -> None: ...


@runtime_checkable
class SequenceCheck(Protocol):
    """One rule about a whole frame sequence."""

    name: str

    def verify(self, sequence: FrameSequence) -> None: ...
```

Drop the unused `field` import if your editor flags it.

- [ ] **Step 6: Write `app/domain/checks/quality.py`**

Every message string below is copied verbatim from `ensure_face_quality`. Do not reformat the specifiers.

```python
from dataclasses import dataclass

from app.domain.checks.base import Frame
from app.domain.errors import ImageQualityError


@dataclass(frozen=True)
class MinSizeCheck:
    """Reject faces whose shorter bbox side is below the floor."""

    min_side_px: float
    name: str = "min_side"

    def verify(self, frame: Frame) -> None:
        side = frame.min_side
        if side < self.min_side_px:
            raise ImageQualityError(
                f"image quality failed: face too small "
                f"(min side={side:.1f}px < {self.min_side_px:.1f}px)"
            )


@dataclass(frozen=True)
class FaceCropCheck:
    """Reject a zero-area crop.

    Unconditional: the legacy gate cropped before any threshold branch, so
    this fires even with every other threshold relaxed to off. Must stay in
    the chain always and must run before any check that reads frame.gray —
    cv2.cvtColor on an empty array raises cv2.error, not a domain error.
    """

    name: str = "face_crop"

    def verify(self, frame: Frame) -> None:
        if frame.crop.size == 0:
            raise ImageQualityError("image quality failed: empty face crop")


@dataclass(frozen=True)
class BlurCheck:
    min_variance: float
    name: str = "blur"

    def verify(self, frame: Frame) -> None:
        variance = frame.metrics.laplacian_variance(frame.gray)
        if variance < self.min_variance:
            raise ImageQualityError(
                f"image quality failed: blur too high "
                f"(laplacian_var={variance:.2f} < {self.min_variance:.2f})"
            )


@dataclass(frozen=True)
class BrightnessMinCheck:
    minimum: float
    name: str = "brightness_min"

    def verify(self, frame: Frame) -> None:
        mean = frame.metrics.mean_brightness(frame.gray)
        if mean < self.minimum:
            raise ImageQualityError(
                f"image quality failed: brightness too low "
                f"(mean={mean:.1f} < {self.minimum:.1f})"
            )


@dataclass(frozen=True)
class BrightnessMaxCheck:
    maximum: float
    name: str = "brightness_max"

    def verify(self, frame: Frame) -> None:
        mean = frame.metrics.mean_brightness(frame.gray)
        if mean > self.maximum:
            raise ImageQualityError(
                f"image quality failed: brightness too high "
                f"(mean={mean:.1f} > {self.maximum:.1f})"
            )
```

- [ ] **Step 7: Write `app/domain/checks/liveness.py` and `motion.py`**

`app/domain/checks/liveness.py`:

```python
from dataclasses import dataclass

from app.domain.checks.base import Frame, FrameSequence
from app.domain.errors import DependencyUnavailableError, LivenessFailedError

MODEL_UNAVAILABLE = (
    "liveness model not available "
    "(run scripts/download_liveness_model.py)"
)


@dataclass(frozen=True)
class LivenessReadyCheck:
    """Probe the scorer before touching it.

    UnavailableLivenessChecker.score_bgr raises RuntimeError, which would
    surface as a 500; this converts the condition into a 503 with the
    legacy detail string.
    """

    name: str = "liveness_ready"

    def verify(self, frame: Frame) -> None:
        if frame.scorer is None or not frame.scorer.ready():
            raise DependencyUnavailableError(MODEL_UNAVAILABLE)


@dataclass(frozen=True)
class LivenessScoreCheck:
    """Single-frame anti-spoof threshold."""

    threshold: float
    name: str = "liveness_score"

    def verify(self, frame: Frame) -> None:
        score = frame.liveness_score
        if score < self.threshold:
            raise LivenessFailedError(
                f"liveness check failed (score={score:.4f} < {self.threshold:.4f})"
            )


@dataclass(frozen=True)
class MeanLivenessCheck:
    """Multi-frame anti-spoof threshold on the mean score."""

    threshold: float
    name: str = "mean_liveness"

    def verify(self, sequence: FrameSequence) -> None:
        mean_score = sequence.mean_liveness
        if mean_score < self.threshold:
            raise LivenessFailedError(
                f"liveness check failed (mean score={mean_score:.4f} "
                f"< {self.threshold:.4f})"
            )
```

`app/domain/checks/motion.py`:

```python
from dataclasses import dataclass

from app.domain.checks.base import FrameSequence
from app.domain.errors import DependencyUnavailableError, LivenessFailedError

NO_KEYPOINTS = (
    "planarity check needs facial keypoints, which this model pack "
    "does not provide (set LIVENESS_MIN_PLANARITY=0 to disable)"
)


@dataclass(frozen=True)
class InterFrameDistanceCheck:
    """Reject a sequence of near-identical frames (a still image replayed)."""

    min_distance: float
    name: str = "inter_frame_distance"

    def verify(self, sequence: FrameSequence) -> None:
        max_distance = sequence.max_pairwise_distance
        if max_distance < self.min_distance:
            raise LivenessFailedError(
                f"liveness check failed (frames too similar: "
                f"distance={max_distance:.4f} < {self.min_distance:.4f})"
            )


@dataclass(frozen=True)
class KeypointsAvailableCheck:
    """The planarity pair needs landmarks from every frame."""

    name: str = "keypoints_available"

    def verify(self, sequence: FrameSequence) -> None:
        if any(kps is None for kps in sequence.keypoints):
            raise DependencyUnavailableError(NO_KEYPOINTS)


@dataclass(frozen=True)
class ViewChangeCheck:
    """Require enough head turn for the parallax score to carry information."""

    min_view_change: float
    name: str = "view_change"

    def verify(self, sequence: FrameSequence) -> None:
        spread = sequence.view_change_spread
        if spread < self.min_view_change:
            raise LivenessFailedError(
                f"liveness check failed (not enough head movement: "
                f"view change={spread:.4f} < {self.min_view_change:.4f})"
            )


@dataclass(frozen=True)
class PlanarityCheck:
    """Reject a flat replay: a real nose shows motion parallax, paper does not."""

    min_planarity: float
    name: str = "planarity"

    def verify(self, sequence: FrameSequence) -> None:
        residual = sequence.planarity_residual
        if residual < self.min_planarity:
            raise LivenessFailedError(
                f"liveness check failed (face looks flat: "
                f"parallax={residual:.4f} < {self.min_planarity:.4f})"
            )
```

- [ ] **Step 8: Write `app/domain/checks/chains.py`**

```python
from app.domain.checks.base import FrameCheck, SequenceCheck
from app.domain.checks.liveness import (
    LivenessReadyCheck,
    LivenessScoreCheck,
    MeanLivenessCheck,
)
from app.domain.checks.motion import (
    InterFrameDistanceCheck,
    KeypointsAvailableCheck,
    PlanarityCheck,
    ViewChangeCheck,
)
from app.domain.checks.quality import (
    BlurCheck,
    BrightnessMaxCheck,
    BrightnessMinCheck,
    FaceCropCheck,
    MinSizeCheck,
)


def build_quality_chain(gate) -> list[FrameCheck]:
    """Image-quality checks in the legacy ensure_face_quality order.

    `gate` is an application.params.GateParams. Typed loosely to keep the
    domain layer from importing the application layer; only attribute reads
    happen here.
    """
    chain: list[FrameCheck] = []
    if gate.min_side_px > 0:
        chain.append(MinSizeCheck(min_side_px=gate.min_side_px))
    chain.append(FaceCropCheck())  # always: the legacy gate cropped unconditionally
    if gate.blur_min_variance > 0:
        chain.append(BlurCheck(min_variance=gate.blur_min_variance))
    if gate.brightness_min > 0:
        chain.append(BrightnessMinCheck(minimum=gate.brightness_min))
    if gate.brightness_max < 255:
        chain.append(BrightnessMaxCheck(maximum=gate.brightness_max))
    return chain


def build_single_frame_chain(gate) -> list[FrameCheck]:
    """Quality, then the model-ready probe, then the single-frame threshold."""
    return [
        *build_quality_chain(gate),
        LivenessReadyCheck(),
        LivenessScoreCheck(threshold=gate.liveness_threshold),
    ]


def planarity_enabled(params) -> bool:
    """True when the response must report a planarity_residual number.

    The legacy early return needed BOTH thresholds at or below zero, so a
    config with min_planarity == 0 and min_view_change > 0 still measures
    and reports the residual.
    """
    return params.min_planarity > 0.0 or params.min_view_change > 0.0


def build_sequence_chain(params) -> list[SequenceCheck]:
    """Multi-frame checks in the legacy _run_multiframe_liveness order.

    `params` is an application.params.MultiframeParams.
    """
    chain: list[SequenceCheck] = [
        MeanLivenessCheck(threshold=params.gate.liveness_threshold),
        InterFrameDistanceCheck(min_distance=params.min_distance),
    ]
    if planarity_enabled(params):
        chain.append(KeypointsAvailableCheck())
        chain.append(ViewChangeCheck(min_view_change=params.min_view_change))
        chain.append(PlanarityCheck(min_planarity=params.min_planarity))
    return chain
```

- [ ] **Step 9: Add the Review Focus #3 test**

Append to `tests/test_domain_checks.py`:

```python
def _mf(**over) -> MultiframeParams:
    gate_over = {k: over.pop(k) for k in list(over) if k in GateParams.__annotations__}
    base = dict(max_frames=5, min_distance=0.05, min_planarity=0.06, min_view_change=0.025)
    base.update(over)
    return MultiframeParams(gate=_gate(**gate_over), **base)


# --- Review Focus #3 -------------------------------------------------------
def test_planarity_residual_reported_when_only_view_change_enabled():
    """min_planarity == 0 does NOT disable the measurement; min_view_change > 0 keeps it on."""
    params = _mf(min_planarity=0.0, min_view_change=0.025)
    assert planarity_enabled(params) is True
    names = [c.name for c in build_sequence_chain(params)]
    assert names == [
        "mean_liveness", "inter_frame_distance",
        "keypoints_available", "view_change", "planarity",
    ]


def test_planarity_disabled_only_when_both_thresholds_are_zero():
    params = _mf(min_planarity=0.0, min_view_change=0.0)
    assert planarity_enabled(params) is False
    assert [c.name for c in build_sequence_chain(params)] == [
        "mean_liveness", "inter_frame_distance",
    ]


def test_keypoints_check_message_is_verbatim():
    from app.domain.checks.motion import KeypointsAvailableCheck

    image = np.full((100, 100, 3), 128, dtype=np.uint8)
    sequence = FrameSequence(frames=[
        _frame(image, [0.0, 0.0, 90.0, 90.0], kps=None),
        _frame(image, [0.0, 0.0, 90.0, 90.0], kps=None),
    ])
    with pytest.raises(DependencyUnavailableError) as exc:
        KeypointsAvailableCheck().verify(sequence)
    assert str(exc.value) == (
        "planarity check needs facial keypoints, which this model pack "
        "does not provide (set LIVENESS_MIN_PLANARITY=0 to disable)"
    )


def test_liveness_ready_check_rejects_unavailable_checker():
    from app.domain.checks.liveness import LivenessReadyCheck
    from app.infrastructure.pipeline.fake_liveness import UnavailableLivenessChecker

    image = np.full((100, 100, 3), 128, dtype=np.uint8)
    frame = _frame(image, [0.0, 0.0, 90.0, 90.0], scorer=UnavailableLivenessChecker())
    with pytest.raises(DependencyUnavailableError) as exc:
        LivenessReadyCheck().verify(frame)
    assert str(exc.value) == (
        "liveness model not available (run scripts/download_liveness_model.py)"
    )


def test_single_frame_liveness_threshold_message_is_verbatim():
    from app.domain.checks.liveness import LivenessScoreCheck
    from app.infrastructure.pipeline.fake_liveness import FakeLivenessChecker

    image = np.full((100, 100, 3), 128, dtype=np.uint8)
    frame = _frame(image, [0.0, 0.0, 90.0, 90.0],
                   scorer=FakeLivenessChecker(score=0.10))
    with pytest.raises(LivenessFailedError) as exc:
        LivenessScoreCheck(threshold=0.5).verify(frame)
    assert str(exc.value) == "liveness check failed (score=0.1000 < 0.5000)"


def test_liveness_score_is_computed_once_per_frame():
    class CountingScorer:
        def __init__(self) -> None:
            self.calls = 0

        def ready(self) -> bool:
            return True

        def score_bgr(self, image_bgr, bbox=None) -> float:
            self.calls += 1
            return 0.9

    image = np.full((100, 100, 3), 128, dtype=np.uint8)
    scorer = CountingScorer()
    frame = _frame(image, [0.0, 0.0, 90.0, 90.0], scorer=scorer)
    for check in build_single_frame_chain(_gate()):
        check.verify(frame)
    assert frame.liveness_score == pytest.approx(0.9)
    assert scorer.calls == 1, "cached_property must not re-run inference"
```

`GateParams` and `MultiframeParams` land in Task 5; this test file imports them. Create `app/application/params.py` as part of this task's Step 10 so the test can run — that is the one piece of the application layer this task needs.

- [ ] **Step 10: Create `app/application/params.py`**

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class GateParams:
    """Resolved single-frame thresholds. Built by the API layer from Settings
    plus per-request overrides, so the application layer never sees Settings."""

    blur_min_variance: float
    brightness_min: float
    brightness_max: float
    min_side_px: float
    liveness_threshold: float


@dataclass(frozen=True)
class MultiframeParams:
    """Resolved multi-frame thresholds."""

    gate: GateParams
    max_frames: int
    min_distance: float
    min_planarity: float
    min_view_change: float
```

Also create empty `app/application/__init__.py`.

- [ ] **Step 11: Reduce `image_quality.py` to a shim over the chain**

`app/infrastructure/pipeline/image_quality.py` re-exports `_crop_face_bgr` from `metrics.py` (Step 4) and replaces the body of `ensure_face_quality` with a chain run, so the legacy call sites in `routes.py` keep working until Task 9 deletes them:

```python
def ensure_face_quality(
    image_bgr,
    bbox,
    *,
    blur_min_variance: float,
    brightness_min: float,
    brightness_max: float,
    min_side_px: float,
) -> None:
    """Deprecated. Runs the domain quality chain; kept for the legacy routes."""
    from app.application.params import GateParams
    from app.domain.checks.base import Frame
    from app.domain.checks.chains import build_quality_chain
    from app.domain.face import DetectedFace
    from app.infrastructure.pipeline.metrics import OpenCvFaceImageMetrics

    gate = GateParams(
        blur_min_variance=blur_min_variance,
        brightness_min=brightness_min,
        brightness_max=brightness_max,
        min_side_px=min_side_px,
        liveness_threshold=0.0,
    )
    frame = Frame(
        image_bgr=image_bgr,
        detected=DetectedFace(embedding=None, bbox=bbox, kps=None),
        metrics=OpenCvFaceImageMetrics(),
    )
    for check in build_quality_chain(gate):
        check.verify(frame)
```

The imports are function-local to avoid a cycle with `metrics.py`'s import of `_crop_face_bgr`. Task 11 deletes this function.

- [ ] **Step 12: Run the check tests to verify they pass**

Run: `python -m pytest tests/test_domain_checks.py -v`
Expected: every test PASSED. If `test_min_size_check_message_is_verbatim` fails on the number format, compare against the legacy f-string character by character rather than adjusting the test.

- [ ] **Step 13: Run the whole suite, including the existing quality tests**

Run: `python -m pytest -q`
Expected: baseline count + new tests, zero failures. The pre-existing image-quality tests now exercise the chain through the shim — if any of them fails, the chain's order or a message string is wrong, not the test.

- [ ] **Step 14: Commit**

```bash
graphify update .
git add app/domain tests/test_domain_checks.py app/application app/infrastructure/pipeline
git commit -m "refactor: replace quality/liveness branches with check chains (OCP)

Eleven one-rule classes behind FrameCheck/SequenceCheck protocols. Splitting
the protocol by arity keeps sequence checks from ignoring a frame argument.
OpenCV measurements move behind a FaceImageMetrics port so domain stays pure.
Error strings and evaluation order are unchanged; ensure_face_quality is now
a shim over the chain."
```

---

### Task 5: `app/application/` — image loader, single-frame gate, enroll & check

The first two use cases, plus the two collaborators every later single-frame use case reuses. This is where Review Focus #1 and #5 get pinned.

**Files:**
- Modify: `app/application/params.py` (add `ImageLimits` and the request dataclasses)
- Create: `app/application/image_loader.py`, `app/application/gate.py`, `app/application/results.py`
- Create: `app/application/use_cases/__init__.py`, `app/application/use_cases/enroll_face.py`, `app/application/use_cases/check_face.py`
- Test: `tests/test_application_single_frame.py`

**Interfaces:**
- Consumes: `ImageDecoder`, `FaceDetector`, `FaceWriter`, `IdentityFactory`, `LivenessScorer`, `FaceImageMetrics` (Tasks 2, 4); `GateParams` (Task 4); `Frame`, `build_single_frame_chain` (Task 4).
- Produces:
  - `ImageLimits(max_side: int, max_bytes: int)`
  - `SingleFrameRequest(image_base64: str, liveness: bool, limits: ImageLimits, gate: GateParams)`
  - `Base64ImageLoader(decoder).load(image_base64, *, limits) -> np.ndarray`
  - `FaceGate(metrics, scorer).run(image_bgr, detected, *, liveness, gate) -> float | None`
  - `EnrollResult(face_id: str, liveness_score: float | None)`, `CheckResult(liveness_score: float | None)`
  - `EnrollFaceUseCase(loader, detector, gate, writer, identity).execute(request: SingleFrameRequest) -> EnrollResult`
  - `CheckFaceUseCase(loader, detector, gate).execute(request: SingleFrameRequest) -> CheckResult`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_application_single_frame.py`:

```python
import base64
import inspect

import numpy as np
import pytest

from app.application.gate import FaceGate
from app.application.image_loader import Base64ImageLoader
from app.application.params import GateParams, ImageLimits, SingleFrameRequest
from app.application.use_cases.check_face import CheckFaceUseCase
from app.application.use_cases.enroll_face import EnrollFaceUseCase
from app.domain.errors import DependencyUnavailableError, InvalidImageError
from app.domain.face import DetectedFace
from app.infrastructure.imaging import OpenCvImageDecoder
from app.infrastructure.pipeline.fake_liveness import FakeLivenessChecker, UnavailableLivenessChecker
from app.infrastructure.pipeline.metrics import OpenCvFaceImageMetrics

LIMITS = ImageLimits(max_side=640, max_bytes=5_000_000)
STRICT_GATE = GateParams(
    blur_min_variance=50.0,
    brightness_min=40.0,
    brightness_max=220.0,
    min_side_px=80.0,
    liveness_threshold=0.5,
)


def _png_base64() -> str:
    """A 200x200 mid-grey PNG: decodes fine, fails every quality gate."""
    import cv2

    image = np.full((200, 200, 3), 128, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", image)
    assert ok
    return base64.b64encode(buf.tobytes()).decode()


class StubDetector:
    def __init__(self, bbox=(0.0, 0.0, 10.0, 10.0)) -> None:
        self.bbox = bbox
        self.calls = 0

    def detect_bgr(self, image_bgr) -> DetectedFace:
        self.calls += 1
        return DetectedFace(
            embedding=np.ones(512, dtype=np.float32),
            bbox=np.asarray(self.bbox, dtype=np.float32),
            kps=None,
        )


class RecordingWriter:
    def __init__(self) -> None:
        self.upserts: list[tuple[str, np.ndarray]] = []

    def upsert(self, face_id: str, vector: np.ndarray) -> None:
        self.upserts.append((face_id, vector))

    def delete(self, face_id: str) -> bool:
        return True


class FixedIdentity:
    def new_id(self) -> str:
        return "fixed-id"


def _enroll(*, scorer, gate=STRICT_GATE):
    writer = RecordingWriter()
    use_case = EnrollFaceUseCase(
        loader=Base64ImageLoader(OpenCvImageDecoder()),
        detector=StubDetector(),
        gate=FaceGate(metrics=OpenCvFaceImageMetrics(), scorer=scorer),
        writer=writer,
        identity=FixedIdentity(),
    )
    return use_case, writer, gate


# --- Review Focus #1 -------------------------------------------------------
def test_enroll_with_liveness_false_runs_no_checks():
    """liveness=false must skip quality AND anti-spoof, matching today.

    The stub detector reports a 10x10 bbox on a flat grey image, which fails
    min_side, blur and brightness. With liveness=false none of that runs.
    """
    use_case, writer, gate = _enroll(scorer=UnavailableLivenessChecker())
    request = SingleFrameRequest(
        image_base64=_png_base64(), liveness=False, limits=LIMITS, gate=gate
    )
    result = use_case.execute(request)
    assert result.face_id == "fixed-id"
    assert result.liveness_score is None
    assert len(writer.upserts) == 1


# --- Review Focus #5 -------------------------------------------------------
def test_unavailable_liveness_checker_raises_dependency_unavailable():
    """ready() is probed before scoring, so RuntimeError never escapes."""
    use_case, _, gate = _enroll(scorer=UnavailableLivenessChecker())
    relaxed = GateParams(
        blur_min_variance=0.0, brightness_min=0.0, brightness_max=255.0,
        min_side_px=0.0, liveness_threshold=0.5,
    )
    request = SingleFrameRequest(
        image_base64=_png_base64(), liveness=True, limits=LIMITS, gate=relaxed
    )
    with pytest.raises(DependencyUnavailableError) as exc:
        use_case.execute(request)
    assert str(exc.value) == (
        "liveness model not available (run scripts/download_liveness_model.py)"
    )


def test_enroll_with_liveness_true_returns_the_score():
    use_case, writer, _ = _enroll(scorer=FakeLivenessChecker(score=0.91))
    relaxed = GateParams(
        blur_min_variance=0.0, brightness_min=0.0, brightness_max=255.0,
        min_side_px=0.0, liveness_threshold=0.5,
    )
    request = SingleFrameRequest(
        image_base64=_png_base64(), liveness=True, limits=LIMITS, gate=relaxed
    )
    result = use_case.execute(request)
    assert result.liveness_score == pytest.approx(0.91)
    assert writer.upserts[0][0] == "fixed-id"


def test_check_face_takes_no_writer_and_returns_no_score_without_liveness():
    """`/v1/faces/check` validates; it must never be able to persist.

    Enrollment's RecordingWriter test (above) covers the case where a writer
    *is* wired in. Here the guarantee is structural: CheckFaceUseCase cannot
    write because no writer can be handed to it. Asserting on the constructor
    signature pins that, where probing a private attribute would not.
    """
    params = inspect.signature(CheckFaceUseCase.__init__).parameters
    assert not [p for p in params if "writer" in p or "store" in p], (
        f"CheckFaceUseCase must take no writer/store collaborator, got {list(params)}"
    )
    use_case = CheckFaceUseCase(
        loader=Base64ImageLoader(OpenCvImageDecoder()),
        detector=StubDetector(),
        gate=FaceGate(metrics=OpenCvFaceImageMetrics(), scorer=FakeLivenessChecker()),
    )
    request = SingleFrameRequest(
        image_base64=_png_base64(), liveness=False, limits=LIMITS, gate=STRICT_GATE
    )
    assert use_case.execute(request).liveness_score is None


def test_loader_strips_a_data_url_prefix():
    loader = Base64ImageLoader(OpenCvImageDecoder())
    payload = "data:image/png;base64," + _png_base64()
    image = loader.load(payload, limits=LIMITS)
    assert image.shape[2] == 3


def test_loader_rejects_non_base64_with_the_legacy_message():
    loader = Base64ImageLoader(OpenCvImageDecoder())
    with pytest.raises(InvalidImageError) as exc:
        loader.load("!!!not base64!!!", limits=LIMITS)
    assert str(exc.value) == "invalid image_base64"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/test_application_single_frame.py -v`
Expected: collection error, `No module named 'app.application.gate'`.

- [ ] **Step 3: Extend `app/application/params.py`**

Append:

```python
@dataclass(frozen=True)
class ImageLimits:
    """Decode guards. Mirrors Settings.max_image_side / max_upload_bytes."""

    max_side: int
    max_bytes: int


@dataclass(frozen=True)
class SingleFrameRequest:
    """One image plus every threshold already resolved against Settings."""

    image_base64: str
    liveness: bool
    limits: ImageLimits
    gate: GateParams


@dataclass(frozen=True)
class VerifyRequest(SingleFrameRequest):
    match_threshold: float


@dataclass(frozen=True)
class SearchRequest(SingleFrameRequest):
    limit: int
    min_score: float
```

`VerifyRequest` and `SearchRequest` are consumed in Task 7; they live here so the whole resolved-parameter vocabulary is in one file.

- [ ] **Step 4: Write `app/application/image_loader.py`**

```python
import base64

import numpy as np

from app.application.params import ImageLimits
from app.domain.errors import InvalidImageError
from app.domain.ports.imaging import ImageDecoder


class Base64ImageLoader:
    """Base64 (optionally data-URL wrapped) -> BGR ndarray.

    Lifted verbatim from the old routes._decode_base64_image, with the
    Settings lookup replaced by an ImageLimits argument.
    """

    def __init__(self, decoder: ImageDecoder) -> None:
        self._decoder = decoder

    def load(self, image_base64: str, *, limits: ImageLimits) -> np.ndarray:
        raw = image_base64.strip()
        if raw.startswith("data:") and "," in raw:
            raw = raw.split(",", 1)[1]
        try:
            data = base64.b64decode(raw, validate=False)
        except Exception as exc:
            raise InvalidImageError("invalid image_base64") from exc
        return self._decoder.decode(
            data, max_side=limits.max_side, max_bytes=limits.max_bytes
        )
```

- [ ] **Step 5: Write `app/application/gate.py`**

```python
from dataclasses import dataclass

import numpy as np

from app.application.params import GateParams
from app.domain.checks.base import Frame
from app.domain.checks.chains import build_single_frame_chain
from app.domain.face import DetectedFace
from app.domain.ports.liveness import LivenessScorer
from app.domain.ports.metrics import FaceImageMetrics


@dataclass(frozen=True)
class FaceGate:
    """Runs the single-frame chain and returns the anti-spoof score.

    Returns None without running any check when liveness is off — the
    behaviour of the old _run_quality_and_liveness. Quality gating is
    deliberately coupled to the liveness flag; do not decouple them here.
    """

    metrics: FaceImageMetrics
    scorer: LivenessScorer

    def run(
        self,
        image_bgr: np.ndarray,
        detected: DetectedFace,
        *,
        liveness: bool,
        gate: GateParams,
    ) -> float | None:
        if not liveness:
            return None
        frame = Frame(
            image_bgr=image_bgr,
            detected=detected,
            metrics=self.metrics,
            scorer=self.scorer,
        )
        for check in build_single_frame_chain(gate):
            check.verify(frame)
        return frame.liveness_score
```

- [ ] **Step 6: Write `app/application/results.py`**

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class EnrollResult:
    face_id: str
    liveness_score: float | None


@dataclass(frozen=True)
class CheckResult:
    liveness_score: float | None


@dataclass(frozen=True)
class UpdateResult:
    face_id: str
    liveness_score: float | None


@dataclass(frozen=True)
class VerifyResult:
    face_id: str
    match: bool
    score: float
    threshold: float
    liveness_score: float | None


@dataclass(frozen=True)
class SearchResult:
    hits: list[tuple[str, float]]
    threshold: float
    liveness_score: float | None


@dataclass(frozen=True)
class LivenessResult:
    mean_score: float
    frame_count: int
    inter_frame_distance: float
    planarity_residual: float | None
```

- [ ] **Step 7: Write the two use cases**

`app/application/use_cases/enroll_face.py`:

```python
from app.application.gate import FaceGate
from app.application.image_loader import Base64ImageLoader
from app.application.params import SingleFrameRequest
from app.application.results import EnrollResult
from app.domain.ports.identity import IdentityFactory
from app.domain.ports.pipeline import FaceDetector
from app.domain.ports.store import FaceWriter


class EnrollFaceUseCase:
    """Decode -> detect -> gate -> mint id -> store."""

    def __init__(
        self,
        *,
        loader: Base64ImageLoader,
        detector: FaceDetector,
        gate: FaceGate,
        writer: FaceWriter,
        identity: IdentityFactory,
    ) -> None:
        self._loader = loader
        self._detector = detector
        self._gate = gate
        self._writer = writer
        self._identity = identity

    def execute(self, request: SingleFrameRequest) -> EnrollResult:
        image = self._loader.load(request.image_base64, limits=request.limits)
        detected = self._detector.detect_bgr(image)
        score = self._gate.run(
            image, detected, liveness=request.liveness, gate=request.gate
        )
        face_id = self._identity.new_id()
        self._writer.upsert(face_id, detected.embedding)
        return EnrollResult(face_id=face_id, liveness_score=score)
```

`app/application/use_cases/check_face.py`:

```python
from app.application.gate import FaceGate
from app.application.image_loader import Base64ImageLoader
from app.application.params import SingleFrameRequest
from app.application.results import CheckResult
from app.domain.ports.pipeline import FaceDetector


class CheckFaceUseCase:
    """Preflight: the same gates as enroll, with no store write."""

    def __init__(
        self,
        *,
        loader: Base64ImageLoader,
        detector: FaceDetector,
        gate: FaceGate,
    ) -> None:
        self._loader = loader
        self._detector = detector
        self._gate = gate

    def execute(self, request: SingleFrameRequest) -> CheckResult:
        image = self._loader.load(request.image_base64, limits=request.limits)
        detected = self._detector.detect_bgr(image)
        score = self._gate.run(
            image, detected, liveness=request.liveness, gate=request.gate
        )
        return CheckResult(liveness_score=score)
```

Both constructors are keyword-only; the test in Step 1 uses keyword arguments throughout. Create `app/application/use_cases/__init__.py` empty.

- [ ] **Step 8: Run the tests to verify they pass**

Run: `python -m pytest tests/test_application_single_frame.py -v`
Expected: all PASSED.

Run: `python -m pytest -q`
Expected: baseline + new tests. `routes.py` is untouched, so the OpenAPI contract stays green.

- [ ] **Step 9: Commit**

```bash
graphify update .
git add app/application tests/test_application_single_frame.py
git commit -m "feat(application): add image loader, single-frame gate, enroll/check use cases

Pins the two behaviours easiest to lose in this refactor: liveness=false
skips every check, and an unavailable checker yields DependencyUnavailableError
rather than a leaked RuntimeError."
```

---

### Task 6: `CheckLivenessUseCase` — the multi-frame path

The 74-line `_run_multiframe_liveness` becomes a 20-line orchestration over the sequence chain. The `max_frames` guard moves from an inline `HTTPException(422)` to a domain error, so the application layer stops importing FastAPI.

**Files:**
- Modify: `app/domain/errors.py` (add `TooManyFramesError`)
- Modify: `app/application/params.py` (add `MultiframeRequest`)
- Create: `app/application/use_cases/check_liveness.py`
- Test: `tests/test_application_multiframe.py`

**Interfaces:**
- Consumes: `Base64ImageLoader`, `FaceGate` collaborators (Task 5); `FrameSequence`, `build_sequence_chain`, `planarity_enabled`, `build_quality_chain` (Task 4); `MultiframeParams` (Task 4).
- Produces:
  - `app.domain.errors.TooManyFramesError` — mapped to **422** by the handler in Task 9.
  - `MultiframeRequest(images_base64: list[str], limits: ImageLimits, params: MultiframeParams)`
  - `CheckLivenessUseCase(loader, detector, metrics, scorer).execute(request) -> LivenessResult`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_application_multiframe.py`:

```python
import base64

import numpy as np
import pytest

from app.application.image_loader import Base64ImageLoader
from app.application.params import GateParams, ImageLimits, MultiframeParams, MultiframeRequest
from app.application.use_cases.check_liveness import CheckLivenessUseCase
from app.domain.errors import DependencyUnavailableError, LivenessFailedError, TooManyFramesError
from app.domain.face import DetectedFace
from app.infrastructure.imaging import OpenCvImageDecoder
from app.infrastructure.pipeline.fake_liveness import FakeLivenessChecker, UnavailableLivenessChecker
from app.infrastructure.pipeline.metrics import OpenCvFaceImageMetrics

LIMITS = ImageLimits(max_side=640, max_bytes=5_000_000)
RELAXED_GATE = GateParams(
    blur_min_variance=0.0, brightness_min=0.0, brightness_max=255.0,
    min_side_px=0.0, liveness_threshold=0.5,
)


def _params(**over) -> MultiframeParams:
    base = dict(
        gate=RELAXED_GATE, max_frames=5, min_distance=0.0,
        min_planarity=0.0, min_view_change=0.0,
    )
    base.update(over)
    return MultiframeParams(**base)


def _png_base64(value: int = 128) -> str:
    import cv2

    ok, buf = cv2.imencode(".png", np.full((200, 200, 3), value, dtype=np.uint8))
    assert ok
    return base64.b64encode(buf.tobytes()).decode()


class SpreadDetector:
    """Returns a different embedding per call so inter-frame distance is > 0.

    `kps` is either a single (5, 2) array shared by every frame, or a list of
    them — one per call — so a test can give each frame distinct landmarks.
    """

    def __init__(self, kps=None) -> None:
        self.index = 0
        self.kps = kps

    def _kps_for_this_call(self):
        if isinstance(self.kps, list):
            return self.kps[self.index % len(self.kps)]
        return self.kps

    def detect_bgr(self, image_bgr) -> DetectedFace:
        vector = np.zeros(512, dtype=np.float32)
        vector[self.index % 512] = 1.0
        kps = self._kps_for_this_call()
        self.index += 1
        return DetectedFace(
            embedding=vector,
            bbox=np.asarray([0.0, 0.0, 100.0, 100.0], dtype=np.float32),
            kps=kps,
        )


def _use_case(*, scorer=None, kps=None) -> CheckLivenessUseCase:
    return CheckLivenessUseCase(
        loader=Base64ImageLoader(OpenCvImageDecoder()),
        detector=SpreadDetector(kps=kps),
        metrics=OpenCvFaceImageMetrics(),
        scorer=scorer or FakeLivenessChecker(score=0.9),
    )


def test_rejects_more_frames_than_the_limit():
    use_case = _use_case()
    request = MultiframeRequest(
        images_base64=[_png_base64()] * 6, limits=LIMITS, params=_params(max_frames=5)
    )
    with pytest.raises(TooManyFramesError) as exc:
        use_case.execute(request)
    assert str(exc.value) == "too many frames (max 5)"


def test_probes_the_scorer_before_decoding_anything():
    use_case = _use_case(scorer=UnavailableLivenessChecker())
    request = MultiframeRequest(
        images_base64=["not-an-image", "also-not"], limits=LIMITS, params=_params()
    )
    with pytest.raises(DependencyUnavailableError) as exc:
        use_case.execute(request)
    assert str(exc.value) == (
        "liveness model not available (run scripts/download_liveness_model.py)"
    )


def test_frame_errors_are_prefixed_with_the_frame_index():
    use_case = _use_case()
    request = MultiframeRequest(
        images_base64=[_png_base64(), "!!!"], limits=LIMITS, params=_params()
    )
    with pytest.raises(Exception) as exc:
        use_case.execute(request)
    assert str(exc.value).startswith("frame[1]: ")


def test_rejects_frames_that_are_too_similar():
    class SameDetector:
        def detect_bgr(self, image_bgr) -> DetectedFace:
            return DetectedFace(
                embedding=np.ones(512, dtype=np.float32),
                bbox=np.asarray([0.0, 0.0, 100.0, 100.0], dtype=np.float32),
                kps=None,
            )

    use_case = CheckLivenessUseCase(
        loader=Base64ImageLoader(OpenCvImageDecoder()),
        detector=SameDetector(),
        metrics=OpenCvFaceImageMetrics(),
        scorer=FakeLivenessChecker(score=0.9),
    )
    request = MultiframeRequest(
        images_base64=[_png_base64(), _png_base64(200)],
        limits=LIMITS,
        params=_params(min_distance=0.05),
    )
    with pytest.raises(LivenessFailedError) as exc:
        use_case.execute(request)
    assert str(exc.value) == (
        "liveness check failed (frames too similar: distance=0.0000 < 0.0500)"
    )


def test_rejects_a_low_mean_score():
    use_case = _use_case(scorer=FakeLivenessChecker(score=0.10))
    request = MultiframeRequest(
        images_base64=[_png_base64(), _png_base64(200)], limits=LIMITS, params=_params()
    )
    with pytest.raises(LivenessFailedError) as exc:
        use_case.execute(request)
    assert str(exc.value) == "liveness check failed (mean score=0.1000 < 0.5000)"


def test_returns_none_residual_when_both_motion_thresholds_are_zero():
    use_case = _use_case()
    request = MultiframeRequest(
        images_base64=[_png_base64(), _png_base64(200)], limits=LIMITS, params=_params()
    )
    result = use_case.execute(request)
    assert result.frame_count == 2
    assert result.mean_score == pytest.approx(0.9)
    assert result.inter_frame_distance > 0.0
    assert result.planarity_residual is None


# --- Review Focus #3 -------------------------------------------------------
def test_multiframe_response_reports_residual_with_planarity_disabled():
    """min_planarity == 0 but min_view_change > 0 must still report a number."""
    # The two frames carry DIFFERENT kps: ViewChangeCheck needs a nonzero
    # view_change_spread to pass, and identical keypoints across frames give
    # a spread of exactly 0.0 (max - min over one repeated value), which
    # would make this check raise before the assertion below ever runs.
    # kps_frontal has equal eye/mouth-corner widths (no foreshortening);
    # kps_turned narrows the interocular distance to simulate a head turn,
    # so view_change (log(interocular/vertical)) differs between the two.
    kps_frontal = np.asarray(
        [[30.0, 40.0], [70.0, 40.0], [50.0, 60.0], [35.0, 80.0], [65.0, 80.0]],
        dtype=np.float32,
    )
    kps_turned = np.asarray(
        [[38.0, 40.0], [62.0, 40.0], [50.0, 60.0], [35.0, 80.0], [65.0, 80.0]],
        dtype=np.float32,
    )
    use_case = _use_case(kps=[kps_frontal, kps_turned])
    request = MultiframeRequest(
        images_base64=[_png_base64(), _png_base64(200)],
        limits=LIMITS,
        params=_params(min_planarity=0.0, min_view_change=0.0001),
    )
    result = use_case.execute(request)
    assert result.planarity_residual is not None
    assert isinstance(result.planarity_residual, float)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/test_application_multiframe.py -v`
Expected: collection error on `app.application.use_cases.check_liveness`.

- [ ] **Step 3: Add `TooManyFramesError` to the domain**

Append to `app/domain/errors.py`:

```python
class TooManyFramesError(Exception):
    """Raised when a liveness request carries more frames than configured."""
```

Add it to `app/errors.py`'s re-export list too.

- [ ] **Step 4: Add `MultiframeRequest` to `app/application/params.py`**

```python
@dataclass(frozen=True)
class MultiframeRequest:
    images_base64: list[str]
    limits: ImageLimits
    params: MultiframeParams
```

- [ ] **Step 5: Write `app/application/use_cases/check_liveness.py`**

```python
from app.application.image_loader import Base64ImageLoader
from app.application.params import MultiframeRequest
from app.application.results import LivenessResult
from app.domain.checks.base import Frame, FrameSequence
from app.domain.checks.chains import (
    build_quality_chain,
    build_sequence_chain,
    planarity_enabled,
)
from app.domain.checks.liveness import MODEL_UNAVAILABLE
from app.domain.errors import (
    DependencyUnavailableError,
    FaceDetectionError,
    InvalidImageError,
    TooManyFramesError,
)
from app.domain.ports.liveness import LivenessScorer
from app.domain.ports.metrics import FaceImageMetrics
from app.domain.ports.pipeline import FaceDetector


class CheckLivenessUseCase:
    """Passive multi-frame liveness.

    Order is load-bearing and copied from the old _run_multiframe_liveness:
    ready() probe, frame-count guard, then per frame decode -> detect ->
    quality chain -> score, then the sequence chain.
    """

    def __init__(
        self,
        *,
        loader: Base64ImageLoader,
        detector: FaceDetector,
        metrics: FaceImageMetrics,
        scorer: LivenessScorer,
    ) -> None:
        self._loader = loader
        self._detector = detector
        self._metrics = metrics
        self._scorer = scorer

    def execute(self, request: MultiframeRequest) -> LivenessResult:
        if not self._scorer.ready():
            raise DependencyUnavailableError(MODEL_UNAVAILABLE)
        if len(request.images_base64) > request.params.max_frames:
            raise TooManyFramesError(
                f"too many frames (max {request.params.max_frames})"
            )

        quality_chain = build_quality_chain(request.params.gate)
        frames: list[Frame] = []
        for index, image_base64 in enumerate(request.images_base64):
            try:
                image = self._loader.load(image_base64, limits=request.limits)
            except InvalidImageError as exc:
                raise InvalidImageError(f"frame[{index}]: {exc}") from exc
            try:
                detected = self._detector.detect_bgr(image)
            except FaceDetectionError as exc:
                raise FaceDetectionError(f"frame[{index}]: {exc}", code=exc.code) from exc
            frame = Frame(
                image_bgr=image,
                detected=detected,
                metrics=self._metrics,
                scorer=self._scorer,
            )
            for check in quality_chain:
                check.verify(frame)
            frame.liveness_score  # force scoring here, in the legacy order
            frames.append(frame)

        sequence = FrameSequence(frames=frames)
        for check in build_sequence_chain(request.params):
            check.verify(sequence)

        return LivenessResult(
            mean_score=sequence.mean_liveness,
            frame_count=len(frames),
            inter_frame_distance=sequence.max_pairwise_distance,
            planarity_residual=(
                sequence.planarity_residual
                if planarity_enabled(request.params)
                else None
            ),
        )
```

The bare `frame.liveness_score` statement is intentional: it triggers the `cached_property` so inference happens inside the loop, in the same position as the legacy `checker.score_bgr` call, rather than later when `FrameSequence.liveness_scores` is first read. Keep the comment; a linter may flag the expression as useless.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `python -m pytest tests/test_application_multiframe.py -v`
Expected: all PASSED.

- [ ] **Step 7: Run the whole suite**

Run: `python -m pytest -q`
Expected: baseline + new tests, zero failures.

- [ ] **Step 8: Commit**

```bash
graphify update .
git add app/domain/errors.py app/errors.py app/application tests/test_application_multiframe.py
git commit -m "feat(application): add CheckLivenessUseCase over the sequence chain

Replaces the 74-line _run_multiframe_liveness. The max_frames guard becomes
TooManyFramesError instead of an inline HTTPException, so the application
layer no longer imports FastAPI. planarity_residual stays a number whenever
either motion threshold is enabled."
```

---

### Task 7: Remaining face use cases + the face/system query services

Five more use cases and two of the three query services. Every one is a direct lift of a current route body with `Settings` and `Depends` removed.

**Files:**
- Create: `app/application/use_cases/verify_face.py`, `search_faces.py`, `update_face.py`, `delete_face.py`, `clear_faces.py`
- Create: `app/application/queries/__init__.py`, `app/application/queries/faces.py`, `app/application/queries/system.py`
- Test: `tests/test_application_faces.py`

**Interfaces:**
- Consumes: `FaceReader`, `FaceWriter`, `FaceSearcher`, `FaceCatalog`, `HealthProbe`, `PipelineProbe` (Task 2); `FaceGate`, `Base64ImageLoader`, `SingleFrameRequest`, `VerifyRequest`, `SearchRequest`, results (Task 5); `app.domain.matching.cosine.cosine_similarity`, `is_match`.
- Produces:
  - `VerifyFaceUseCase(loader, detector, gate, reader).execute(face_id: str, request: VerifyRequest) -> VerifyResult`
  - `SearchFacesUseCase(loader, detector, gate, searcher).execute(request: SearchRequest) -> SearchResult`
  - `UpdateFaceUseCase(loader, detector, gate, reader, writer).execute(face_id: str, request: SingleFrameRequest) -> UpdateResult`
  - `DeleteFaceUseCase(writer).execute(face_id: str) -> None`
  - `ClearFacesUseCase(catalog).execute() -> int`
  - `FaceQueryService(reader, catalog)` — `count() -> int`, `ensure_exists(face_id: str) -> None`, `list_ids(*, limit: int, offset: int) -> tuple[list[str], int]`
  - `SystemQueryService(pipeline_probe, store_probe, catalog)` — `is_healthy() -> bool`, `stats() -> tuple[int, bool, bool]` returning `(faces_count, store_ok, pipeline_ok)`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_application_faces.py`:

```python
import numpy as np
import pytest

from app.application.queries.faces import FaceQueryService
from app.application.queries.system import SystemQueryService
from app.application.use_cases.clear_faces import ClearFacesUseCase
from app.application.use_cases.delete_face import DeleteFaceUseCase
from app.domain.errors import FaceNotFoundError
from app.infrastructure.store.memory import MemoryFaceStore


def _store_with(ids: list[str]) -> MemoryFaceStore:
    store = MemoryFaceStore()
    for index, face_id in enumerate(ids):
        vector = np.zeros(512, dtype=np.float32)
        vector[index % 512] = 1.0
        store.upsert(face_id, vector)
    return store


def test_face_query_service_counts_and_lists():
    store = _store_with(["a", "b", "c"])
    queries = FaceQueryService(reader=store, catalog=store)
    assert queries.count() == 3
    ids, total = queries.list_ids(limit=2, offset=0)
    assert len(ids) == 2
    assert total == 3


def test_ensure_exists_raises_the_legacy_message():
    store = _store_with(["a"])
    queries = FaceQueryService(reader=store, catalog=store)
    queries.ensure_exists("a")
    with pytest.raises(FaceNotFoundError) as exc:
        queries.ensure_exists("missing")
    assert str(exc.value) == "face missing not found"


def test_delete_use_case_raises_when_absent():
    store = _store_with(["a"])
    use_case = DeleteFaceUseCase(writer=store)
    use_case.execute("a")
    with pytest.raises(FaceNotFoundError) as exc:
        use_case.execute("a")
    assert str(exc.value) == "face a not found"


def test_clear_returns_the_deleted_count():
    store = _store_with(["a", "b"])
    assert ClearFacesUseCase(catalog=store).execute() == 2
    assert store.count() == 0


def test_system_queries_report_each_probe_independently():
    class DownProbe:
        def ready(self) -> bool:
            return False

    store = _store_with(["a"])
    queries = SystemQueryService(
        pipeline_probe=DownProbe(), store_probe=store, catalog=store
    )
    assert queries.is_healthy() is False
    faces_count, store_ok, pipeline_ok = queries.stats()
    assert (faces_count, store_ok, pipeline_ok) == (1, True, False)
```

Confirm `MemoryFaceStore.clear()` returns the deleted count (`graphify explain "MemoryFaceStore"`); if it returns `None`, the legacy `ClearResponse(deleted_count=store.clear())` already depended on that, so leave the adapter alone and keep the test asserting whatever `clear()` actually returns.

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/test_application_faces.py -v`
Expected: collection error on `app.application.queries.faces`.

- [ ] **Step 3: Write the two query services**

`app/application/queries/faces.py`:

```python
from app.domain.errors import FaceNotFoundError
from app.domain.ports.store import FaceCatalog, FaceReader


class FaceQueryService:
    """Read-only face lookups. No gating, no writes, no liveness."""

    def __init__(self, *, reader: FaceReader, catalog: FaceCatalog) -> None:
        self._reader = reader
        self._catalog = catalog

    def count(self) -> int:
        return self._catalog.count()

    def ensure_exists(self, face_id: str) -> None:
        if self._reader.get(face_id) is None:
            raise FaceNotFoundError(f"face {face_id} not found")

    def list_ids(self, *, limit: int, offset: int) -> tuple[list[str], int]:
        return (
            self._catalog.list_ids(limit=limit, offset=offset),
            self._catalog.count(),
        )
```

`app/application/queries/system.py`:

```python
from app.domain.ports.pipeline import PipelineProbe
from app.domain.ports.store import FaceCatalog, HealthProbe


class SystemQueryService:
    """Backs /health and /v1/stats."""

    def __init__(
        self,
        *,
        pipeline_probe: PipelineProbe,
        store_probe: HealthProbe,
        catalog: FaceCatalog,
    ) -> None:
        self._pipeline_probe = pipeline_probe
        self._store_probe = store_probe
        self._catalog = catalog

    def is_healthy(self) -> bool:
        return self._pipeline_probe.ready() and self._store_probe.ping()

    def stats(self) -> tuple[int, bool, bool]:
        return (
            self._catalog.count(),
            self._store_probe.ping(),
            self._pipeline_probe.ready(),
        )
```

`stats()` calls `count()`, `ping()`, `ready()` in that order, matching the legacy `StatsResponse(...)` construction.

- [ ] **Step 4: Write the five use cases**

`app/application/use_cases/verify_face.py`:

```python
from app.application.gate import FaceGate
from app.application.image_loader import Base64ImageLoader
from app.application.params import VerifyRequest
from app.application.results import VerifyResult
from app.domain.errors import FaceNotFoundError
from app.domain.matching.cosine import cosine_similarity, is_match
from app.domain.ports.pipeline import FaceDetector
from app.domain.ports.store import FaceReader


class VerifyFaceUseCase:
    """1:1 match against one enrolled embedding."""

    def __init__(
        self,
        *,
        loader: Base64ImageLoader,
        detector: FaceDetector,
        gate: FaceGate,
        reader: FaceReader,
    ) -> None:
        self._loader = loader
        self._detector = detector
        self._gate = gate
        self._reader = reader

    def execute(self, face_id: str, request: VerifyRequest) -> VerifyResult:
        stored = self._reader.get(face_id)
        if stored is None:
            raise FaceNotFoundError(f"face {face_id} not found")
        image = self._loader.load(request.image_base64, limits=request.limits)
        detected = self._detector.detect_bgr(image)
        live_score = self._gate.run(
            image, detected, liveness=request.liveness, gate=request.gate
        )
        score = cosine_similarity(stored, detected.embedding)
        return VerifyResult(
            face_id=face_id,
            match=is_match(score, request.match_threshold),
            score=score,
            threshold=request.match_threshold,
            liveness_score=live_score,
        )
```

The store lookup happens **before** decoding, matching the legacy route: a missing `face_id` returns 404 without paying for a decode.

`app/application/use_cases/search_faces.py`:

```python
from app.application.gate import FaceGate
from app.application.image_loader import Base64ImageLoader
from app.application.params import SearchRequest
from app.application.results import SearchResult
from app.domain.ports.pipeline import FaceDetector
from app.domain.ports.store import FaceSearcher


class SearchFacesUseCase:
    """1:N search over the vector store."""

    def __init__(
        self,
        *,
        loader: Base64ImageLoader,
        detector: FaceDetector,
        gate: FaceGate,
        searcher: FaceSearcher,
    ) -> None:
        self._loader = loader
        self._detector = detector
        self._gate = gate
        self._searcher = searcher

    def execute(self, request: SearchRequest) -> SearchResult:
        image = self._loader.load(request.image_base64, limits=request.limits)
        detected = self._detector.detect_bgr(image)
        live_score = self._gate.run(
            image, detected, liveness=request.liveness, gate=request.gate
        )
        hits = self._searcher.search(
            detected.embedding, limit=request.limit, min_score=request.min_score
        )
        return SearchResult(
            hits=hits, threshold=request.min_score, liveness_score=live_score
        )
```

`app/application/use_cases/update_face.py`:

```python
from app.application.gate import FaceGate
from app.application.image_loader import Base64ImageLoader
from app.application.params import SingleFrameRequest
from app.application.results import UpdateResult
from app.domain.errors import FaceNotFoundError
from app.domain.ports.pipeline import FaceDetector
from app.domain.ports.store import FaceReader, FaceWriter


class UpdateFaceUseCase:
    """Re-enroll an existing face_id with a new image."""

    def __init__(
        self,
        *,
        loader: Base64ImageLoader,
        detector: FaceDetector,
        gate: FaceGate,
        reader: FaceReader,
        writer: FaceWriter,
    ) -> None:
        self._loader = loader
        self._detector = detector
        self._gate = gate
        self._reader = reader
        self._writer = writer

    def execute(self, face_id: str, request: SingleFrameRequest) -> UpdateResult:
        if self._reader.get(face_id) is None:
            raise FaceNotFoundError(f"face {face_id} not found")
        image = self._loader.load(request.image_base64, limits=request.limits)
        detected = self._detector.detect_bgr(image)
        live_score = self._gate.run(
            image, detected, liveness=request.liveness, gate=request.gate
        )
        self._writer.upsert(face_id, detected.embedding)
        return UpdateResult(face_id=face_id, liveness_score=live_score)
```

`app/application/use_cases/delete_face.py`:

```python
from app.domain.errors import FaceNotFoundError
from app.domain.ports.store import FaceWriter


class DeleteFaceUseCase:
    def __init__(self, *, writer: FaceWriter) -> None:
        self._writer = writer

    def execute(self, face_id: str) -> None:
        if not self._writer.delete(face_id):
            raise FaceNotFoundError(f"face {face_id} not found")
```

`app/application/use_cases/clear_faces.py`:

```python
from app.domain.ports.store import FaceCatalog


class ClearFacesUseCase:
    def __init__(self, *, catalog: FaceCatalog) -> None:
        self._catalog = catalog

    def execute(self) -> int:
        return self._catalog.clear()
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_application_faces.py -v`
Expected: all PASSED.

Run: `python -m pytest -q`
Expected: baseline + new tests.

- [ ] **Step 6: Commit**

```bash
graphify update .
git add app/application tests/test_application_faces.py
git commit -m "feat(application): add verify/search/update/delete/clear use cases and face queries

Writes go through use cases; reads go through query services. Each depends
only on the store roles it actually calls."
```

---

### Task 8: Group use cases, queries, and the `RebuildLock` seam

`GroupRebuildService` currently owns four things: a `threading.Lock`, session lifecycle, orchestration, and failure persistence. The lock and the session move behind ports; the orchestration stays where it is.

The TOCTOU race between `has_running()` and `create_run()` is **left open on purpose** — see the spec's "Known limitation". Do not add a constraint or widen the lock.

**Files:**
- Modify: `app/jobs/group_rebuild.py` → `app/infrastructure/jobs/group_rebuild.py` (takes a `RebuildLock` and a `UnitOfWork` instead of constructing its own)
- Create: `app/application/use_cases/start_group_rebuild.py`
- Create: `app/application/queries/groups.py`
- Create: `app/application/params.py` additions: `RebuildParams`, `GroupsListParams`
- Create: `app/application/results.py` additions: `GroupSnapshotView`, `RebuildRunView`
- Test: `tests/test_application_groups.py`

**Interfaces:**
- Consumes: `RebuildLock`, `UnitOfWork`, `GroupsReader`, `GroupsWriter`, `FaceCatalog`, `FaceSearcher` (Tasks 2, 3).
- Produces:
  - `RebuildParams(min_score: float | None, min_size: int | None, top_k: int | None)` — passed through to `GroupRebuildService.start` unchanged, which keeps resolving its own defaults from `Settings`.
  - `GroupsListParams(min_size: int, fallback_threshold: float)`
  - `app.domain.errors.GroupRebuildConflictError`, `app.domain.errors.InvalidGroupSizeError`, `app.domain.errors.GroupRunNotFoundError`
  - `StartGroupRebuildUseCase(service).execute(*, store, settings, params) -> str` returning the `run_id`
  - `GroupsQueryService(unit_of_work)` — `latest_groups(params) -> GroupSnapshotView`, `list_runs() -> list[RebuildRunView]`, `get_run(run_id: str) -> RebuildRunView`
  - `GroupSnapshotView(threshold: float, groups: list[tuple[list[str], int]], run_id: str | None, built_at: datetime | None)`
  - `RebuildRunView` mirroring the fields `_group_rebuild_status` read: `run_id, status, threshold, top_k, min_size, face_count, group_count, error, started_at, finished_at`

`StartGroupRebuildUseCase.execute` still receives `settings` because `GroupRebuildService.start` resolves `face_search_min_score`, `face_groups_min_size` and `face_groups_ann_top_k` from it. Threading `Settings` through this one call is the deliberate exception recorded in the spec; do not refactor `GroupRebuildService`'s own signature in this task.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_application_groups.py`:

```python
import pytest

from app.application.params import GroupsListParams
from app.application.queries.groups import GroupsQueryService
from app.domain.errors import InvalidGroupSizeError
from app.infrastructure.locks.thread_lock import ThreadRebuildLock


class FakeRepo:
    def __init__(self, snapshot=None, runs=None) -> None:
        self._snapshot = snapshot
        self._runs = runs or []

    def latest_completed(self):
        return self._snapshot

    def list_runs(self):
        return self._runs

    def get_run(self, run_id):
        return next((r for r in self._runs if r.id == run_id), None)


class FakeUnitOfWork:
    def __init__(self, repo: FakeRepo) -> None:
        self._repo = repo
        self.opened = 0

    def groups(self):
        from contextlib import contextmanager

        @contextmanager
        def _cm():
            self.opened += 1
            yield self._repo

        return _cm()


def test_empty_snapshot_falls_back_to_the_configured_threshold():
    uow = FakeUnitOfWork(FakeRepo(snapshot=None))
    view = GroupsQueryService(unit_of_work=uow).latest_groups(
        GroupsListParams(min_size=2, fallback_threshold=0.40)
    )
    assert view.threshold == pytest.approx(0.40)
    assert view.groups == []
    assert view.run_id is None
    assert view.built_at is None
    assert uow.opened == 1


def test_groups_smaller_than_min_size_are_filtered_out():
    class Group:
        def __init__(self, face_ids, size):
            self.face_ids = face_ids
            self.size = size

    class Snapshot:
        threshold = 0.5
        run_id = "run-1"
        built_at = None
        groups = [Group(["a", "b"], 2), Group(["c", "d", "e"], 3)]

    uow = FakeUnitOfWork(FakeRepo(snapshot=Snapshot()))
    view = GroupsQueryService(unit_of_work=uow).latest_groups(
        GroupsListParams(min_size=3, fallback_threshold=0.40)
    )
    assert [size for _, size in view.groups] == [3]
    assert view.threshold == pytest.approx(0.5)
    assert view.run_id == "run-1"


def test_get_run_raises_the_legacy_not_found_message():
    uow = FakeUnitOfWork(FakeRepo(runs=[]))
    from app.domain.errors import GroupRunNotFoundError

    with pytest.raises(GroupRunNotFoundError) as exc:
        GroupsQueryService(unit_of_work=uow).get_run("nope")
    assert str(exc.value) == "group run nope not found"


def test_min_size_below_two_is_rejected():
    with pytest.raises(InvalidGroupSizeError) as exc:
        GroupsListParams.validated(min_size=1, fallback_threshold=0.4)
    assert str(exc.value) == "min_size must be >= 2"


def test_rebuild_lock_blocks_a_second_concurrent_start():
    lock = ThreadRebuildLock()
    assert lock.acquire(blocking=False) is True
    assert lock.acquire(blocking=False) is False
    lock.release()
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/test_application_groups.py -v`
Expected: collection error on `app.application.queries.groups`.

- [ ] **Step 3: Add the three group errors to the domain**

Append to `app/domain/errors.py` and to `app/errors.py`'s re-export list:

```python
class GroupRebuildConflictError(Exception):
    """Raised when a rebuild is already in flight. Maps to 409."""


class InvalidGroupSizeError(Exception):
    """Raised when a caller asks for groups smaller than 2. Maps to 400."""


class GroupRunNotFoundError(Exception):
    """Raised when a rebuild run_id is unknown. Maps to 404."""
```

`GroupRebuildConflict` already exists in `app/jobs/group_rebuild.py`. Keep that class and make it subclass `GroupRebuildConflictError`, so the existing `except GroupRebuildConflict` in the legacy route and the new exception handler both work:

```python
class GroupRebuildConflict(GroupRebuildConflictError):
    """..."""  # keep the existing docstring
```

- [ ] **Step 4: Add the params and views**

Append to `app/application/params.py`:

```python
@dataclass(frozen=True)
class RebuildParams:
    """Client overrides, passed through to GroupRebuildService.start unchanged."""

    min_score: float | None
    min_size: int | None
    top_k: int | None


@dataclass(frozen=True)
class GroupsListParams:
    min_size: int
    fallback_threshold: float

    @classmethod
    def validated(cls, *, min_size: int, fallback_threshold: float) -> "GroupsListParams":
        from app.domain.errors import InvalidGroupSizeError

        if min_size < 2:
            raise InvalidGroupSizeError("min_size must be >= 2")
        return cls(min_size=min_size, fallback_threshold=fallback_threshold)
```

Append to `app/application/results.py`:

```python
@dataclass(frozen=True)
class GroupSnapshotView:
    threshold: float
    groups: list[tuple[list[str], int]]
    run_id: str | None
    built_at: datetime | None


@dataclass(frozen=True)
class RebuildRunView:
    run_id: str
    status: str
    threshold: float
    top_k: int
    min_size: int
    face_count: int | None
    group_count: int | None
    error: str | None
    started_at: datetime | None
    finished_at: datetime | None
```

Add `from datetime import datetime` at the top of `results.py`. The views carry `datetime`, not ISO strings — formatting is the API layer's job (Task 9's `_iso`).

- [ ] **Step 5: Write `app/application/queries/groups.py`**

```python
from app.application.params import GroupsListParams
from app.application.results import GroupSnapshotView, RebuildRunView
from app.domain.errors import GroupRunNotFoundError
from app.domain.ports.unit_of_work import UnitOfWork


def _to_view(run) -> RebuildRunView:
    return RebuildRunView(
        run_id=run.id,
        status=run.status.value,
        threshold=run.threshold,
        top_k=run.top_k,
        min_size=run.min_size,
        face_count=run.face_count,
        group_count=run.group_count,
        error=run.error,
        started_at=run.started_at,
        finished_at=run.finished_at,
    )


class GroupsQueryService:
    """Read-only group and rebuild-run lookups.

    Replaces the three routes that opened a session by hand; the unit of work
    commits/rolls back/closes for them.
    """

    def __init__(self, *, unit_of_work: UnitOfWork) -> None:
        self._uow = unit_of_work

    def latest_groups(self, params: GroupsListParams) -> GroupSnapshotView:
        with self._uow.groups() as repo:
            snapshot = repo.latest_completed()
            if snapshot is None:
                return GroupSnapshotView(
                    threshold=params.fallback_threshold,
                    groups=[],
                    run_id=None,
                    built_at=None,
                )
            return GroupSnapshotView(
                threshold=snapshot.threshold,
                groups=[
                    (g.face_ids, g.size)
                    for g in snapshot.groups
                    if g.size >= params.min_size
                ],
                run_id=snapshot.run_id,
                built_at=snapshot.built_at,
            )

    def list_runs(self) -> list[RebuildRunView]:
        with self._uow.groups() as repo:
            return [_to_view(run) for run in repo.list_runs()]

    def get_run(self, run_id: str) -> RebuildRunView:
        with self._uow.groups() as repo:
            run = repo.get_run(run_id)
            if run is None:
                raise GroupRunNotFoundError(f"group run {run_id} not found")
            return _to_view(run)
```

- [ ] **Step 6: Write `app/application/use_cases/start_group_rebuild.py`**

```python
from app.application.params import RebuildParams


class StartGroupRebuildUseCase:
    """Claims the single-flight slot and creates the run row.

    The actual rebuild runs later, via the service's execute() method handed
    to FastAPI's BackgroundTasks by the router. Settings is threaded through
    because GroupRebuildService.start still resolves its own defaults from it;
    see the spec's SRP deviation note.
    """

    def __init__(self, *, service) -> None:
        self._service = service

    def execute(self, *, store, settings, params: RebuildParams) -> str:
        return self._service.start(
            store=store,
            settings=settings,
            min_score=params.min_score,
            min_size=params.min_size,
            top_k=params.top_k,
        )
```

- [ ] **Step 7: Move `GroupRebuildService` and inject the lock and unit of work**

```bash
mkdir -p app/infrastructure/jobs
touch app/infrastructure/jobs/__init__.py
git mv app/jobs/group_rebuild.py app/infrastructure/jobs/group_rebuild.py
```

Change the constructor from

```python
def __init__(self, session_factory) -> None:
    self._session_factory = session_factory
    self._lock = threading.Lock()
```

to

```python
def __init__(self, session_factory, *, lock=None) -> None:
    self._session_factory = session_factory
    self._lock = lock if lock is not None else ThreadRebuildLock()
```

and replace the three direct lock call sites — `self._lock.acquire(blocking=False)`, `self._lock.release()`, and `self._lock.locked()` (used in the `finally` branch of `execute()`) — with the same calls against the port; they are already signature-compatible now that `RebuildLock` declares `locked()` too, so the bodies do not change. Import `ThreadRebuildLock` from `app.infrastructure.locks.thread_lock`.

Leave `has_running()` / `create_run()` / `commit()` exactly as they are. Add a comment above them:

```python
        # TOCTOU: another worker process can pass has_running() before this
        # one commits. Deliberate — see the spec's "Known limitation". Closing
        # it needs a distributed RebuildLock adapter, not a change here.
```

Leave `app/jobs/group_rebuild.py` as a re-export shim (`from app.infrastructure.jobs.group_rebuild import GroupRebuildConflict, GroupRebuildService`) so `app/deps.py` and `routes.py` keep importing until Task 9.

- [ ] **Step 8: Run the tests to verify they pass**

Run: `python -m pytest tests/test_application_groups.py -v`
Expected: all PASSED.

Run: `python -m pytest -q`
Expected: baseline + new tests. The group-rebuild integration tests must still pass — the service's behaviour is unchanged.

- [ ] **Step 9: Commit**

```bash
graphify update .
git add app/application app/infrastructure/jobs app/jobs app/domain/errors.py app/errors.py tests/test_application_groups.py
git commit -m "refactor: extract RebuildLock port, add group use case and query service

The three routes that opened a session by hand now go through a UnitOfWork.
The rebuild lock is injected, so a distributed adapter is a wiring change.
The has_running/create_run race is left open and documented, per the spec."
```

---

### Task 9: `app/api/` — four tagged routers, schema split, handlers, providers

`routes.py` is deleted. Each router handler becomes: build params from `Settings` + body, call one use case, shape the response. No `try/except`, no `HTTPException` in a handler body except the two that the legacy code raised directly (`/health` 503 and the rebuild 409/400) — and those move into `exception_handlers.py` so the handlers stay thin.

**Files:**
- Create: `app/api/params_builder.py`
- Create: `app/api/schemas/__init__.py`, `faces.py`, `liveness.py`, `groups.py`, `system.py`
- Create: `app/api/exception_handlers.py`
- Rewrite: `app/api/dependencies.py` (replacing `app/deps.py`)
- Create: `app/api/routers/__init__.py`, `faces.py`, `liveness.py`, `groups.py`, `system.py`
- Delete: `app/api/routes.py`, `app/deps.py`
- Modify: `app/api/schemas.py` → re-export shim over `app/api/schemas/` (one test imports it), `tests/test_api_faces.py` (repoint the `settings_dep` import)
- Test: `tests/test_api_params_builder.py`, `tests/test_route_order.py`

**Interfaces:**
- Consumes: every use case and query service from Tasks 5–8.
- Produces:
  - `app.api.params_builder`: `resolve(client, default)`, `image_limits(settings)`, `gate_params(body, settings)`, `multiframe_params(body, settings)`, `clamp(value, *, default, maximum)`, `iso(dt)`
  - `app.api.dependencies`: one provider per use case / query service, each reading `request.app.state`
  - `app.api.exception_handlers.register(app)` — the full error→status map
  - `app.api.routers.{faces,liveness,groups,system}.router` — `APIRouter(tags=[...])`

- [ ] **Step 1: Split the schemas without changing a single field**

A module and a package cannot share the name `app.api.schemas`, so this is a delete-and-recreate:

```bash
git rm app/api/schemas.py --cached
mkdir -p app/api/schemas
```

Move the 26 models out of the old `app/api/schemas.py` into four modules by tag, keeping every field, default, `Field(...)` constraint and `ConfigDict` byte-identical:

- `app/api/schemas/faces.py` — `FaceImageRequest`, `FaceVerifyRequest`, `FaceSearchRequest`, `FaceListRequest`, `FaceIdResponse`, `CheckResponse`, `UpdateResponse`, `DeleteResponse`, `ClearResponse`, `CountResponse`, `FaceExistsResponse`, `FaceListResponse`, `VerifyResponse`, `SearchHit`, `SearchResponse`
- `app/api/schemas/liveness.py` — `FaceLivenessRequest`, `LivenessResponse`
- `app/api/schemas/groups.py` — `GroupsListRequest`, `GroupRebuildRequest`, `FaceGroup`, `GroupsResponse`, `GroupRebuildAccepted`, `GroupRebuildStatus`, `GroupRebuildListResponse`
- `app/api/schemas/system.py` — `HealthResponse`, `StatsResponse`

Then delete the old module file (`rm app/api/schemas.py`) and have `app/api/schemas/__init__.py` re-export all 26 names, so `from app.api.schemas import X` keeps resolving:

```python
from app.api.schemas.faces import (
    CheckResponse,
    ClearResponse,
    CountResponse,
    DeleteResponse,
    FaceExistsResponse,
    FaceIdResponse,
    FaceImageRequest,
    FaceListRequest,
    FaceListResponse,
    FaceSearchRequest,
    FaceVerifyRequest,
    SearchHit,
    SearchResponse,
    UpdateResponse,
    VerifyResponse,
)
from app.api.schemas.groups import (
    FaceGroup,
    GroupRebuildAccepted,
    GroupRebuildListResponse,
    GroupRebuildRequest,
    GroupRebuildStatus,
    GroupsListRequest,
    GroupsResponse,
)
from app.api.schemas.liveness import FaceLivenessRequest, LivenessResponse
from app.api.schemas.system import HealthResponse, StatsResponse

__all__ = [
    "CheckResponse", "ClearResponse", "CountResponse", "DeleteResponse",
    "FaceExistsResponse", "FaceGroup", "FaceIdResponse", "FaceImageRequest",
    "FaceListRequest", "FaceListResponse", "FaceLivenessRequest",
    "FaceSearchRequest", "FaceVerifyRequest", "GroupRebuildAccepted",
    "GroupRebuildListResponse", "GroupRebuildRequest", "GroupRebuildStatus",
    "GroupsListRequest", "GroupsResponse", "HealthResponse", "LivenessResponse",
    "SearchHit", "SearchResponse", "StatsResponse", "UpdateResponse",
    "VerifyResponse",
]
```

`from app.api.schemas import X` keeps working for the one existing test that uses it. Run `python -m pytest tests/test_openapi_contract.py -v` immediately after this step — component schema names and shapes must be identical.

- [ ] **Step 2: Write the params-builder test**

Create `tests/test_api_params_builder.py`:

```python
import pytest

from app.api.params_builder import clamp, gate_params, image_limits, iso, multiframe_params, resolve
from app.api.schemas import FaceImageRequest, FaceLivenessRequest
from app.config import Settings

SETTINGS = Settings()


def test_resolve_prefers_the_client_value():
    assert resolve(None, 0.5) == pytest.approx(0.5)
    assert resolve(0.1, 0.5) == pytest.approx(0.1)
    assert resolve(0.0, 0.5) == pytest.approx(0.0), "0.0 is a value, not absence"


def test_clamp_matches_the_legacy_limit_arithmetic():
    assert clamp(None, default=100, maximum=1000) == 100
    assert clamp(0, default=100, maximum=1000) == 1
    assert clamp(-5, default=100, maximum=1000) == 1
    assert clamp(5000, default=100, maximum=1000) == 1000
    assert clamp(50, default=100, maximum=1000) == 50


def test_image_limits_come_from_settings():
    limits = image_limits(SETTINGS)
    assert limits.max_side == SETTINGS.max_image_side
    assert limits.max_bytes == SETTINGS.max_upload_bytes


def test_gate_params_fall_back_to_settings():
    body = FaceImageRequest(image_base64="x", liveness=True)
    gate = gate_params(body, SETTINGS)
    assert gate.blur_min_variance == pytest.approx(SETTINGS.face_blur_min_variance)
    assert gate.brightness_min == pytest.approx(SETTINGS.face_brightness_min)
    assert gate.brightness_max == pytest.approx(SETTINGS.face_brightness_max)
    assert gate.min_side_px == pytest.approx(SETTINGS.face_min_side_px)
    assert gate.liveness_threshold == pytest.approx(SETTINGS.liveness_threshold)


def test_gate_params_honour_a_zero_override():
    body = FaceImageRequest(image_base64="x", liveness=True, blur_min_variance=0.0)
    assert gate_params(body, SETTINGS).blur_min_variance == pytest.approx(0.0)


def test_multiframe_params_resolve_all_eight_overrides():
    body = FaceLivenessRequest(
        images_base64=["a", "b"],
        blur_min_variance=1.0, brightness_min=2.0, brightness_max=3.0,
        min_side_px=4.0, liveness_threshold=0.6,
        liveness_min_distance=0.07, liveness_min_planarity=0.08,
        liveness_min_view_change=0.09,
    )
    params = multiframe_params(body, SETTINGS)
    assert params.max_frames == SETTINGS.liveness_max_frames
    assert params.min_distance == pytest.approx(0.07)
    assert params.min_planarity == pytest.approx(0.08)
    assert params.min_view_change == pytest.approx(0.09)
    assert params.gate.liveness_threshold == pytest.approx(0.6)


def test_iso_passes_none_through():
    from datetime import datetime

    assert iso(None) is None
    assert iso(datetime(2026, 10, 28, 12, 0)) == "2026-10-28T12:00:00"
```

- [ ] **Step 3: Run it to verify it fails**

Run: `python -m pytest tests/test_api_params_builder.py -v`
Expected: collection error on `app.api.params_builder`.

- [ ] **Step 4: Write `app/api/params_builder.py`**

```python
"""Settings + request body -> frozen application params.

The only place in the codebase that reads both Settings and a pydantic body.
Every function here is a lift of a private helper from the old routes.py.
"""

from datetime import datetime

from app.application.params import GateParams, ImageLimits, MultiframeParams
from app.config import Settings


def resolve(client: float | None, default: float) -> float:
    """None means "not supplied"; 0.0 means "disable this check"."""
    return default if client is None else client


def clamp(value: int | None, *, default: int, maximum: int) -> int:
    """Legacy limit arithmetic: default on None, then max(1, min(v, maximum))."""
    effective = default if value is None else value
    return max(1, min(effective, maximum))


def iso(dt: datetime | None) -> str | None:
    return None if dt is None else dt.isoformat()


def image_limits(settings: Settings) -> ImageLimits:
    return ImageLimits(
        max_side=settings.max_image_side, max_bytes=settings.max_upload_bytes
    )


def gate_params(body, settings: Settings) -> GateParams:
    """`body` is any schema carrying the five single-frame overrides."""
    return GateParams(
        blur_min_variance=resolve(
            body.blur_min_variance, settings.face_blur_min_variance
        ),
        brightness_min=resolve(body.brightness_min, settings.face_brightness_min),
        brightness_max=resolve(body.brightness_max, settings.face_brightness_max),
        min_side_px=resolve(body.min_side_px, settings.face_min_side_px),
        liveness_threshold=resolve(
            body.liveness_threshold, settings.liveness_threshold
        ),
    )


def multiframe_params(body, settings: Settings) -> MultiframeParams:
    return MultiframeParams(
        gate=gate_params(body, settings),
        max_frames=settings.liveness_max_frames,
        min_distance=resolve(
            body.liveness_min_distance, settings.liveness_min_distance
        ),
        min_planarity=resolve(
            body.liveness_min_planarity, settings.liveness_min_planarity
        ),
        min_view_change=resolve(
            body.liveness_min_view_change, settings.liveness_min_view_change
        ),
    )
```

- [ ] **Step 5: Write `app/api/exception_handlers.py`**

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.domain.errors import (
    DependencyUnavailableError,
    FaceDetectionError,
    FaceNotFoundError,
    GroupRebuildConflictError,
    GroupRunNotFoundError,
    ImageQualityError,
    InvalidGroupSizeError,
    InvalidImageError,
    LivenessFailedError,
    TooManyFramesError,
)

# Single source of truth for error -> HTTP status. Lifted from
# main._register_exception_handlers plus the five statuses that used to be
# inline HTTPException raises in routes.py.
STATUS_MAP: dict[type[Exception], int] = {
    InvalidImageError: 400,
    InvalidGroupSizeError: 400,
    FaceNotFoundError: 404,
    GroupRunNotFoundError: 404,
    GroupRebuildConflictError: 409,
    FaceDetectionError: 422,
    ImageQualityError: 422,
    LivenessFailedError: 422,
    TooManyFramesError: 422,
    DependencyUnavailableError: 503,
}


def register(app: FastAPI) -> None:
    for exc_type, status in STATUS_MAP.items():
        def handler(request: Request, exc: Exception, _status: int = status):
            return JSONResponse(status_code=_status, content={"detail": str(exc)})

        app.add_exception_handler(exc_type, handler)
```

The `_status=status` default argument is the late-binding guard from the current `main.py` — keep it.

- [ ] **Step 6: Write `app/api/dependencies.py`**

Replaces `app/deps.py`. Keeps the same `app.state` keys and the same `DependencyUnavailableError` messages so the "not configured" responses do not change, and adds a provider per use case.

```python
from fastapi import Depends, Request

from app.application.gate import FaceGate
from app.application.image_loader import Base64ImageLoader
from app.application.queries.faces import FaceQueryService
from app.application.queries.groups import GroupsQueryService
from app.application.queries.system import SystemQueryService
from app.application.use_cases.check_face import CheckFaceUseCase
from app.application.use_cases.check_liveness import CheckLivenessUseCase
from app.application.use_cases.clear_faces import ClearFacesUseCase
from app.application.use_cases.delete_face import DeleteFaceUseCase
from app.application.use_cases.enroll_face import EnrollFaceUseCase
from app.application.use_cases.search_faces import SearchFacesUseCase
from app.application.use_cases.start_group_rebuild import StartGroupRebuildUseCase
from app.application.use_cases.update_face import UpdateFaceUseCase
from app.application.use_cases.verify_face import VerifyFaceUseCase
from app.config import Settings, get_settings
from app.domain.errors import DependencyUnavailableError


def _from_state(request: Request, name: str, message: str):
    value = getattr(request.app.state, name, None)
    if value is None:
        raise DependencyUnavailableError(message)
    return value


def settings_dep() -> Settings:
    return get_settings()


def pipeline_dep(request: Request):
    return _from_state(request, "pipeline", "face pipeline not configured")


def store_dep(request: Request):
    return _from_state(request, "store", "face store not configured")


def liveness_dep(request: Request):
    return _from_state(request, "liveness", "liveness checker not configured")


def metrics_dep(request: Request):
    return _from_state(request, "metrics", "image metrics not configured")


def identity_dep(request: Request):
    return _from_state(request, "identity", "identity factory not configured")


def decoder_dep(request: Request):
    return _from_state(request, "decoder", "image decoder not configured")


def unit_of_work_dep(request: Request):
    return _from_state(request, "unit_of_work", "database not configured")


def session_factory_dep(request: Request):
    return _from_state(request, "session_factory", "database not configured")


def group_rebuild_dep(request: Request):
    return _from_state(request, "group_rebuild", "group rebuild service not configured")


def loader_dep(decoder=Depends(decoder_dep)) -> Base64ImageLoader:
    return Base64ImageLoader(decoder)


def gate_dep(metrics=Depends(metrics_dep), checker=Depends(liveness_dep)) -> FaceGate:
    return FaceGate(metrics=metrics, scorer=checker)


def enroll_dep(
    loader=Depends(loader_dep),
    pipeline=Depends(pipeline_dep),
    gate=Depends(gate_dep),
    store=Depends(store_dep),
    identity=Depends(identity_dep),
) -> EnrollFaceUseCase:
    return EnrollFaceUseCase(
        loader=loader, detector=pipeline, gate=gate, writer=store, identity=identity
    )


def check_face_dep(
    loader=Depends(loader_dep),
    pipeline=Depends(pipeline_dep),
    gate=Depends(gate_dep),
) -> CheckFaceUseCase:
    return CheckFaceUseCase(loader=loader, detector=pipeline, gate=gate)


def verify_dep(
    loader=Depends(loader_dep),
    pipeline=Depends(pipeline_dep),
    gate=Depends(gate_dep),
    store=Depends(store_dep),
) -> VerifyFaceUseCase:
    return VerifyFaceUseCase(
        loader=loader, detector=pipeline, gate=gate, reader=store
    )


def search_dep(
    loader=Depends(loader_dep),
    pipeline=Depends(pipeline_dep),
    gate=Depends(gate_dep),
    store=Depends(store_dep),
) -> SearchFacesUseCase:
    return SearchFacesUseCase(
        loader=loader, detector=pipeline, gate=gate, searcher=store
    )


def update_dep(
    loader=Depends(loader_dep),
    pipeline=Depends(pipeline_dep),
    gate=Depends(gate_dep),
    store=Depends(store_dep),
) -> UpdateFaceUseCase:
    return UpdateFaceUseCase(
        loader=loader, detector=pipeline, gate=gate, reader=store, writer=store
    )


def delete_dep(store=Depends(store_dep)) -> DeleteFaceUseCase:
    return DeleteFaceUseCase(writer=store)


def clear_dep(store=Depends(store_dep)) -> ClearFacesUseCase:
    return ClearFacesUseCase(catalog=store)


def check_liveness_dep(
    loader=Depends(loader_dep),
    pipeline=Depends(pipeline_dep),
    metrics=Depends(metrics_dep),
    checker=Depends(liveness_dep),
) -> CheckLivenessUseCase:
    return CheckLivenessUseCase(
        loader=loader, detector=pipeline, metrics=metrics, scorer=checker
    )


def face_queries_dep(store=Depends(store_dep)) -> FaceQueryService:
    return FaceQueryService(reader=store, catalog=store)


def system_queries_dep(
    pipeline=Depends(pipeline_dep), store=Depends(store_dep)
) -> SystemQueryService:
    return SystemQueryService(
        pipeline_probe=pipeline, store_probe=store, catalog=store
    )


def group_queries_dep(uow=Depends(unit_of_work_dep)) -> GroupsQueryService:
    return GroupsQueryService(unit_of_work=uow)


def start_rebuild_dep(service=Depends(group_rebuild_dep)) -> StartGroupRebuildUseCase:
    return StartGroupRebuildUseCase(service=service)
```

`metrics`, `identity`, `decoder` and `unit_of_work` are new `app.state` keys; Task 10 sets them. `session_factory_dep` is kept because `_wire_database` and the group-rebuild service still use it.

- [ ] **Step 7: Verify the params-builder tests pass**

Run: `python -m pytest tests/test_api_params_builder.py -v`
Expected: all PASSED.

- [ ] **Step 8: Write the route-order test**

Create `tests/test_route_order.py`:

```python
import base64

import numpy as np

from app.bootstrap.app_factory import create_app


def _paths(app) -> list[str]:
    return [r.path for r in app.routes if hasattr(r, "path")]


# --- Review Focus #4 -------------------------------------------------------
def test_faces_count_is_registered_before_face_id():
    paths = _paths(create_app(testing=True))
    assert "/v1/faces/count" in paths
    assert "/v1/faces/{face_id}" in paths
    assert paths.index("/v1/faces/count") < paths.index("/v1/faces/{face_id}"), (
        "GET /v1/faces/count must come first or it is matched as a face_id"
    )


def test_faces_count_not_shadowed_by_face_id_route(client):
    """The behavioural half: count returns a count, not a 404."""
    response = client.get("/v1/faces/count")
    assert response.status_code == 200
    assert "count" in response.json()


def test_groups_rebuild_collection_before_run_id(client):
    assert client.get("/v1/faces/groups/rebuild").status_code == 200
    assert client.get("/v1/faces/groups/rebuild/does-not-exist").status_code == 404


def test_openapi_has_exactly_four_tag_sections(client):
    spec = client.get("/openapi.json").json()
    assert [t["name"] for t in spec["tags"]] == [
        "Faces", "Liveness", "Groups", "System",
    ]
    counts: dict[str, int] = {}
    for methods in spec["paths"].values():
        for op in methods.values():
            for tag in op.get("tags", []):
                counts[tag] = counts.get(tag, 0) + 1
    assert counts == {"Faces": 10, "Liveness": 1, "Groups": 4, "System": 2}
```

`spec["openapi_tags"]` is not a key on the served OpenAPI document — `openapi_tags` is only the FastAPI constructor kwarg (Task 10 passes it as `openapi_tags=OPENAPI_TAGS`); FastAPI renders it into the document's `tags` key. The assertions above read `spec["tags"]` for the order and the per-tag operation counts.

- [ ] **Step 9: Run the route-order test to verify it fails**

Run: `python -m pytest tests/test_route_order.py -v`
Expected: FAIL — `app.bootstrap.app_factory` does not exist yet (Task 10) and no route carries a tag. Leave this test red until Task 10 Step 5; it is the acceptance gate for the two tasks together. Note that in the task log.

- [ ] **Step 10: Write `app/api/routers/faces.py`**

Registration order inside this file is the HTTP matching order. `count` and `list` come before any `{face_id}` route.

```python
from fastapi import APIRouter, Depends

from app.api.dependencies import (
    check_face_dep,
    clear_dep,
    delete_dep,
    enroll_dep,
    face_queries_dep,
    search_dep,
    settings_dep,
    update_dep,
    verify_dep,
)
from app.api.params_builder import clamp, gate_params, image_limits
from app.api.schemas.faces import (
    CheckResponse,
    ClearResponse,
    CountResponse,
    DeleteResponse,
    FaceExistsResponse,
    FaceIdResponse,
    FaceImageRequest,
    FaceListRequest,
    FaceListResponse,
    FaceSearchRequest,
    FaceVerifyRequest,
    SearchHit,
    SearchResponse,
    UpdateResponse,
    VerifyResponse,
)
from app.application.params import SearchRequest, SingleFrameRequest, VerifyRequest
from app.config import Settings

router = APIRouter(tags=["Faces"])


def _single(body: FaceImageRequest, settings: Settings) -> SingleFrameRequest:
    return SingleFrameRequest(
        image_base64=body.image_base64,
        liveness=body.liveness,
        limits=image_limits(settings),
        gate=gate_params(body, settings),
    )


@router.post("", response_model=FaceIdResponse, status_code=201)
def enroll_face(
    body: FaceImageRequest,
    settings: Settings = Depends(settings_dep),
    use_case=Depends(enroll_dep),
) -> FaceIdResponse:
    """Enroll a new face and return its generated face_id."""
    result = use_case.execute(_single(body, settings))
    return FaceIdResponse(face_id=result.face_id, liveness_score=result.liveness_score)


@router.post("/check", response_model=CheckResponse)
def check_face(
    body: FaceImageRequest,
    settings: Settings = Depends(settings_dep),
    use_case=Depends(check_face_dep),
) -> CheckResponse:
    """Preflight: same gates as enroll, no store write."""
    result = use_case.execute(_single(body, settings))
    return CheckResponse(ok=True, liveness_score=result.liveness_score)


@router.delete("", response_model=ClearResponse)
def clear_faces(use_case=Depends(clear_dep)) -> ClearResponse:
    """Remove all enrolled face embeddings (manual full wipe)."""
    return ClearResponse(cleared=True, deleted_count=use_case.execute())


# Static segments first: /count and /list must not be matched as {face_id}.
@router.get("/count", response_model=CountResponse)
def count_faces(queries=Depends(face_queries_dep)) -> CountResponse:
    return CountResponse(count=queries.count())


@router.post("/list", response_model=FaceListResponse)
def list_faces(
    body: FaceListRequest,
    settings: Settings = Depends(settings_dep),
    queries=Depends(face_queries_dep),
) -> FaceListResponse:
    effective = clamp(
        body.limit,
        default=settings.face_list_default_limit,
        maximum=settings.face_list_max_limit,
    )
    face_ids, total = queries.list_ids(limit=effective, offset=body.offset)
    return FaceListResponse(
        face_ids=face_ids, total=total, limit=effective, offset=body.offset
    )


@router.post("/search", response_model=SearchResponse)
def search_faces(
    body: FaceSearchRequest,
    settings: Settings = Depends(settings_dep),
    use_case=Depends(search_dep),
) -> SearchResponse:
    effective_limit = clamp(
        body.limit,
        default=settings.face_search_default_limit,
        maximum=settings.face_search_max_limit,
    )
    threshold = (
        settings.face_search_min_score if body.min_score is None else body.min_score
    )
    base = _single(body, settings)
    result = use_case.execute(
        SearchRequest(
            image_base64=base.image_base64,
            liveness=base.liveness,
            limits=base.limits,
            gate=base.gate,
            limit=effective_limit,
            min_score=threshold,
        )
    )
    return SearchResponse(
        results=[SearchHit(face_id=fid, score=score) for fid, score in result.hits],
        threshold=result.threshold,
        liveness_score=result.liveness_score,
    )


# Dynamic segment routes last.
@router.put("/{face_id}", response_model=UpdateResponse)
def update_face(
    face_id: str,
    body: FaceImageRequest,
    settings: Settings = Depends(settings_dep),
    use_case=Depends(update_dep),
) -> UpdateResponse:
    result = use_case.execute(face_id, _single(body, settings))
    return UpdateResponse(
        face_id=result.face_id, updated=True, liveness_score=result.liveness_score
    )


@router.get("/{face_id}", response_model=FaceExistsResponse)
def get_face(face_id: str, queries=Depends(face_queries_dep)) -> FaceExistsResponse:
    queries.ensure_exists(face_id)
    return FaceExistsResponse(face_id=face_id, exists=True)


@router.delete("/{face_id}", response_model=DeleteResponse)
def delete_face(face_id: str, use_case=Depends(delete_dep)) -> DeleteResponse:
    use_case.execute(face_id)
    return DeleteResponse(face_id=face_id, deleted=True)


@router.post("/{face_id}/verify", response_model=VerifyResponse)
def verify_face(
    face_id: str,
    body: FaceVerifyRequest,
    settings: Settings = Depends(settings_dep),
    use_case=Depends(verify_dep),
) -> VerifyResponse:
    base = _single(body, settings)
    result = use_case.execute(
        face_id,
        VerifyRequest(
            image_base64=base.image_base64,
            liveness=base.liveness,
            limits=base.limits,
            gate=base.gate,
            match_threshold=(
                settings.face_match_threshold
                if body.match_threshold is None
                else body.match_threshold
            ),
        ),
    )
    return VerifyResponse(
        face_id=result.face_id,
        match=result.match,
        score=result.score,
        threshold=result.threshold,
        liveness_score=result.liveness_score,
    )
```

This router is mounted with `prefix="/v1/faces"` in Task 10, which is why the paths here are `""`, `"/check"`, `"/{face_id}"` and so on. Ten operations: enroll, check, clear, count, list, search, update, get, delete, verify.

- [ ] **Step 11: Write `app/api/routers/liveness.py`**

```python
from fastapi import APIRouter, Depends

from app.api.dependencies import check_liveness_dep, settings_dep
from app.api.params_builder import image_limits, multiframe_params
from app.api.schemas.liveness import FaceLivenessRequest, LivenessResponse
from app.application.params import MultiframeRequest
from app.config import Settings

router = APIRouter(tags=["Liveness"])


@router.post("/v1/faces/liveness", response_model=LivenessResponse)
def check_liveness(
    body: FaceLivenessRequest,
    settings: Settings = Depends(settings_dep),
    use_case=Depends(check_liveness_dep),
) -> LivenessResponse:
    """Passive multi-frame liveness: mean score + inter-frame variation."""
    result = use_case.execute(
        MultiframeRequest(
            images_base64=body.images_base64,
            limits=image_limits(settings),
            params=multiframe_params(body, settings),
        )
    )
    return LivenessResponse(
        ok=True,
        liveness_score=result.mean_score,
        frame_count=result.frame_count,
        inter_frame_distance=result.inter_frame_distance,
        planarity_residual=result.planarity_residual,
    )
```

This router declares the full path and is mounted with no prefix, because `/v1/faces/liveness` would otherwise be shadowed by the faces router's `/{face_id}` — it must be registered **before** the faces router in Task 10.

- [ ] **Step 12: Write `app/api/routers/groups.py`**

```python
from fastapi import APIRouter, BackgroundTasks, Depends

from app.api.dependencies import (
    group_queries_dep,
    group_rebuild_dep,
    settings_dep,
    start_rebuild_dep,
    store_dep,
)
from app.api.params_builder import iso
from app.api.schemas.groups import (
    FaceGroup,
    GroupRebuildAccepted,
    GroupRebuildListResponse,
    GroupRebuildRequest,
    GroupRebuildStatus,
    GroupsListRequest,
    GroupsResponse,
)
from app.application.params import GroupsListParams, RebuildParams
from app.application.results import RebuildRunView
from app.config import Settings

router = APIRouter(prefix="/v1/faces/groups", tags=["Groups"])


def _status(view: RebuildRunView) -> GroupRebuildStatus:
    return GroupRebuildStatus(
        run_id=view.run_id,
        status=view.status,
        threshold=view.threshold,
        top_k=view.top_k,
        min_size=view.min_size,
        face_count=view.face_count,
        group_count=view.group_count,
        error=view.error,
        started_at=iso(view.started_at),
        finished_at=iso(view.finished_at),
    )


@router.post("/rebuild", response_model=GroupRebuildAccepted, status_code=202)
def rebuild_face_groups(
    body: GroupRebuildRequest,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(settings_dep),
    store=Depends(store_dep),
    use_case=Depends(start_rebuild_dep),
    service=Depends(group_rebuild_dep),
) -> GroupRebuildAccepted:
    run_id = use_case.execute(
        store=store,
        settings=settings,
        params=RebuildParams(
            min_score=body.min_score, min_size=body.min_size, top_k=body.top_k
        ),
    )
    background_tasks.add_task(service.execute, run_id, store=store, settings=settings)
    return GroupRebuildAccepted(run_id=run_id, status="running")


@router.get("/rebuild", response_model=GroupRebuildListResponse)
def list_group_rebuild_runs(
    queries=Depends(group_queries_dep),
) -> GroupRebuildListResponse:
    return GroupRebuildListResponse(
        runs=[_status(view) for view in queries.list_runs()]
    )


@router.get("/rebuild/{run_id}", response_model=GroupRebuildStatus)
def get_group_rebuild_status(
    run_id: str, queries=Depends(group_queries_dep)
) -> GroupRebuildStatus:
    return _status(queries.get_run(run_id))


@router.post("/list", response_model=GroupsResponse)
def list_face_groups(
    body: GroupsListRequest,
    settings: Settings = Depends(settings_dep),
    queries=Depends(group_queries_dep),
) -> GroupsResponse:
    params = GroupsListParams.validated(
        min_size=(
            settings.face_groups_min_size if body.min_size is None else body.min_size
        ),
        fallback_threshold=settings.face_search_min_score,
    )
    view = queries.latest_groups(params)
    return GroupsResponse(
        threshold=view.threshold,
        groups=[FaceGroup(face_ids=ids, size=size) for ids, size in view.groups],
        run_id=view.run_id,
        built_at=iso(view.built_at),
    )
```

`GroupRebuildConflict` and `ValueError` are no longer caught here: the conflict is a `GroupRebuildConflictError` (409 via the handler) and `GroupRebuildService.start`'s two `ValueError`s need one more change — convert them to `InvalidGroupSizeError` so they still map to 400. Edit `app/infrastructure/jobs/group_rebuild.py`:

```python
        if min_size < 2:
            raise InvalidGroupSizeError("min_size must be >= 2")
        if top_k < 1:
            raise InvalidGroupSizeError("top_k must be >= 1")
```

Both messages are unchanged, so the 400 bodies are unchanged.

- [ ] **Step 13: Write `app/api/routers/system.py`**

```python
from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import system_queries_dep
from app.api.schemas.system import HealthResponse, StatsResponse

router = APIRouter(tags=["System"])


@router.get("/health", response_model=HealthResponse)
def health(queries=Depends(system_queries_dep)) -> HealthResponse:
    if not queries.is_healthy():
        raise HTTPException(status_code=503, detail="dependencies not ready")
    return HealthResponse(status="ok")


@router.get("/v1/stats", response_model=StatsResponse)
def stats(queries=Depends(system_queries_dep)) -> StatsResponse:
    faces_count, store_ok, pipeline_ok = queries.stats()
    return StatsResponse(
        faces_count=faces_count, store_ok=store_ok, pipeline_ok=pipeline_ok
    )
```

This is the one surviving `HTTPException`: `/health` returns `503 {"detail": "dependencies not ready"}` for a *degraded-but-reachable* system, which is not an exceptional condition to raise inward. Keeping it here is the pragmatic exception the spec allows; it stays inside `app/api/`, so no layer rule is broken.

- [ ] **Step 14: Repoint `tests/test_api_faces.py`'s `settings_dep` import**

`settings_dep` is moving from `app/deps.py` (deleted in the next step) to `app/api/dependencies.py` (Step 6). `tests/test_api_faces.py:7` still imports it from the old location and overrides it on `app.dependency_overrides` at line 25 — if this import is not repointed to the same function object the routers now depend on, the override silently does nothing and those tests start hitting real settings.

Before:

```python
from app.deps import settings_dep
```

After:

```python
from app.api.dependencies import settings_dep
```

- [ ] **Step 15: Delete the monolith**

```bash
git rm app/api/routes.py app/deps.py
```

`app/main.py` still imports them — that breaks until Task 10. Do Task 10 before running the suite; the two tasks share one green bar. If you are running tasks with independent review gates, merge Tasks 9 and 10 into a single reviewed unit.

- [ ] **Step 16: Commit (red suite is expected here)**

```bash
git add -A app/api tests/test_api_params_builder.py tests/test_route_order.py tests/test_api_faces.py
git commit -m "refactor(api): split the 541-line routes.py into four tagged routers

Handlers now build params, call one use case, and shape a response. The
error->status map moves to a single table in exception_handlers. Static
segments (/count, /list, /search) stay registered before /{face_id}.

Suite is red until the bootstrap task rewires main.py."
```

---

### Task 10: `app/bootstrap/` — the composition root

`main.py`'s 135 lines split in two: `lifespan.py` builds the production adapters and fails fast; `app_factory.py` assembles the app, registers handlers, and includes the four routers in the right order. `main.py` shrinks to two lines. This task turns the suite green again.

**Files:**
- Create: `app/bootstrap/__init__.py`, `app/bootstrap/lifespan.py`, `app/bootstrap/app_factory.py`
- Rewrite: `app/main.py`
- Test: `tests/test_bootstrap.py`; `tests/test_route_order.py` turns green here

**Interfaces:**
- Consumes: the four routers, `exception_handlers.register`, every infrastructure adapter.
- Produces:
  - `app.bootstrap.lifespan.wire_database(app, database_url) -> None` (the old `_wire_database`, unchanged behaviour including the interrupted-run sweep)
  - `app.bootstrap.lifespan.production_lifespan(app)` — async context manager
  - `app.bootstrap.app_factory.create_app(*, testing: bool = False) -> FastAPI`
  - `app.bootstrap.app_factory.OPENAPI_TAGS` — the four tag metadata dicts, in display order
  - `app.main.app` — module-level instance, so `uvicorn app.main:app` keeps working

- [ ] **Step 1: Write the failing bootstrap test**

Create `tests/test_bootstrap.py`:

```python
import pytest

from app.bootstrap.app_factory import OPENAPI_TAGS, create_app


def test_testing_app_populates_every_state_key():
    app = create_app(testing=True)
    for key in (
        "pipeline", "store", "liveness", "metrics", "identity", "decoder",
        "session_factory", "unit_of_work", "group_rebuild", "db_engine",
    ):
        assert getattr(app.state, key, None) is not None, f"app.state.{key} is unset"


def test_tag_metadata_order_and_names():
    assert [t["name"] for t in OPENAPI_TAGS] == ["Faces", "Liveness", "Groups", "System"]
    assert all(t.get("description") for t in OPENAPI_TAGS), "every tag needs a description"


def test_main_module_exposes_an_app_instance():
    import app.main

    assert app.main.app is not None


def test_missing_state_key_yields_503_not_500():
    from fastapi.testclient import TestClient

    api = create_app(testing=True)
    api.state.store = None
    response = TestClient(api, raise_server_exceptions=False).get("/v1/faces/count")
    assert response.status_code == 503
    assert response.json() == {"detail": "face store not configured"}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_bootstrap.py -v`
Expected: collection error on `app.bootstrap.app_factory`.

- [ ] **Step 3: Write `app/bootstrap/lifespan.py`**

Move `_wire_database` and `_production_lifespan` out of `main.py` verbatim, renamed without the leading underscore, with the import paths updated and the four new `app.state` keys added. The `GroupRun` sweep, the three `try/except` blocks and every log message stay exactly as they are.

```python
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import update

from app.infrastructure.db.models import GroupRun, GroupRunStatus
from app.infrastructure.db.session import create_db_engine, init_db, make_session_factory
from app.infrastructure.db.unit_of_work import SqlAlchemyUnitOfWork
from app.infrastructure.identity import Uuid4IdentityFactory
from app.infrastructure.imaging import OpenCvImageDecoder
from app.infrastructure.jobs.group_rebuild import GroupRebuildService
from app.infrastructure.locks.thread_lock import ThreadRebuildLock
from app.infrastructure.pipeline.metrics import OpenCvFaceImageMetrics

logger = logging.getLogger(__name__)


def wire_stateless_adapters(app: FastAPI) -> None:
    """Adapters with no I/O to set up. Same instances for prod and tests."""
    app.state.metrics = OpenCvFaceImageMetrics()
    app.state.identity = Uuid4IdentityFactory()
    app.state.decoder = OpenCvImageDecoder()


def wire_database(app: FastAPI, database_url: str) -> None:
    engine = create_db_engine(database_url)
    init_db(engine)
    session_factory = make_session_factory(engine)
    app.state.db_engine = engine
    app.state.session_factory = session_factory
    app.state.unit_of_work = SqlAlchemyUnitOfWork(session_factory)
    app.state.group_rebuild = GroupRebuildService(
        session_factory, lock=ThreadRebuildLock()
    )

    # Mark interrupted runs so a restart cannot block rebuild forever.
    session = session_factory()
    try:
        session.execute(
            update(GroupRun)
            .where(GroupRun.status.in_((GroupRunStatus.pending, GroupRunStatus.running)))
            .values(status=GroupRunStatus.failed, error="interrupted by process restart")
        )
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@asynccontextmanager
async def production_lifespan(app: FastAPI):
    # Fail fast with a clear message: the service is useless without models and Qdrant.
    from app.config import get_settings
    from app.infrastructure.pipeline.onnx_insightface import InsightFacePipeline
    from app.infrastructure.store.qdrant import QdrantFaceStore

    settings = get_settings()
    wire_stateless_adapters(app)
    try:
        app.state.pipeline = InsightFacePipeline(
            settings.model_dir, embedding_dim=settings.embedding_dim
        )
    except Exception as exc:
        logger.error(
            "Failed to load InsightFace models from %r (run scripts/download_models.py): %s",
            settings.model_dir,
            exc,
        )
        raise RuntimeError(
            f"InsightFace pipeline init failed (model_dir={settings.model_dir!r}): {exc}"
        ) from exc
    try:
        app.state.store = QdrantFaceStore.from_settings(settings)
    except Exception as exc:
        logger.error("Failed to connect to Qdrant at %s: %s", settings.qdrant_url, exc)
        raise RuntimeError(
            f"Qdrant store init failed (url={settings.qdrant_url!r}): {exc}"
        ) from exc
    try:
        wire_database(app, settings.database_url)
    except Exception as exc:
        logger.error("Failed to connect to database at %s: %s", settings.database_url, exc)
        raise RuntimeError(
            f"Database init failed (url={settings.database_url!r}): {exc}"
        ) from exc
    from app.infrastructure.pipeline.onnx_liveness import load_liveness_checker

    app.state.liveness = load_liveness_checker(
        settings.liveness_model_path, crop_scale=settings.liveness_crop_scale
    )
    if not app.state.liveness.ready():
        logger.warning(
            "Liveness model not loaded from %r — requests with liveness=true will return 503. "
            "Run: python scripts/download_liveness_model.py",
            settings.liveness_model_path,
        )
    yield
```

If the GPU merge from Task 0 changed the `InsightFacePipeline` constructor (e.g. added a `device` argument), copy that call site as it now stands on main — do not revert it.

- [ ] **Step 4: Write `app/bootstrap/app_factory.py`**

```python
from fastapi import FastAPI

from app.api import exception_handlers
from app.api.routers import faces, groups, liveness, system
from app.bootstrap.lifespan import (
    production_lifespan,
    wire_database,
    wire_stateless_adapters,
)

# Swagger section order. Four sections, matching the four routers.
OPENAPI_TAGS = [
    {"name": "Faces", "description": "Enroll, verify, search, update and delete faces."},
    {"name": "Liveness", "description": "Passive multi-frame anti-spoof checks."},
    {"name": "Groups", "description": "Identity clustering and rebuild runs."},
    {"name": "System", "description": "Health and service statistics."},
]


def create_app(*, testing: bool = False) -> FastAPI:
    app = FastAPI(
        title="iM.FaceId",
        version="0.1.0",
        openapi_tags=OPENAPI_TAGS,
        lifespan=None if testing else production_lifespan,
    )
    app.state.testing = testing
    if testing:
        from app.infrastructure.pipeline.fake import FakeFacePipeline
        from app.infrastructure.pipeline.fake_liveness import FakeLivenessChecker
        from app.infrastructure.store.memory import MemoryFaceStore

        wire_stateless_adapters(app)
        app.state.pipeline = FakeFacePipeline()
        app.state.store = MemoryFaceStore()
        app.state.liveness = FakeLivenessChecker()
        wire_database(app, "sqlite+pysqlite:///:memory:")

    exception_handlers.register(app)

    # Registration order is HTTP matching order.
    # 1. liveness: /v1/faces/liveness would be swallowed by faces' /{face_id}.
    # 2. groups:   /v1/faces/groups/* likewise.
    # 3. faces:    its own file keeps /count and /list ahead of /{face_id}.
    # 4. system:   /health and /v1/stats share no prefix with the others.
    app.include_router(liveness.router)
    app.include_router(groups.router)
    app.include_router(faces.router, prefix="/v1/faces")
    app.include_router(system.router)
    return app
```

The ordering comment is the most important four lines in this file. `/v1/faces/liveness` and `/v1/faces/groups/list` are both two-segment-deep paths under `/v1/faces/`, and the faces router's `/{face_id}` plus `/{face_id}/verify` would match them. Registering those two routers first is what keeps them reachable.

- [ ] **Step 5: Rewrite `app/main.py`**

```python
"""ASGI entry point. Wiring lives in app/bootstrap/."""

from app.bootstrap.app_factory import create_app

app = create_app()

__all__ = ["app", "create_app"]
```

`create_app` is re-exported because four existing tests do `from app.main import create_app`.

- [ ] **Step 6: Run the bootstrap and route-order tests**

Run: `python -m pytest tests/test_bootstrap.py tests/test_route_order.py -v`
Expected: all PASSED. If `test_openapi_has_exactly_four_tag_sections` reports the wrong counts, compare the per-tag numbers against the Global Constraints line — a count of 11 for `Faces` usually means the liveness route got the wrong tag.

- [ ] **Step 7: Run the OpenAPI contract test**

Run: `python -m pytest tests/test_openapi_contract.py -v`
Expected: 2 PASSED. The signature helper deliberately ignores `tags`, so adding them does not break the contract. Any other failure here is a real regression — a changed path, status, or response schema. Fix the code, not the baseline.

- [ ] **Step 8: Run the whole suite**

Run: `python -m pytest -q`
Expected: the Task 0 baseline count plus every test added in Tasks 1–10, zero failures. This is the first green bar since Task 9.

- [ ] **Step 9: Commit**

```bash
graphify update .
git add app/bootstrap app/main.py tests/test_bootstrap.py
git commit -m "refactor(bootstrap): extract the composition root from main.py

main.py is now two lines. Router registration order is documented because it
is load-bearing: the liveness and groups routers must precede faces/{face_id}.
Four openapi_tags give Swagger four sections."
```

---

### Task 11: Delete the shims, lock the layer rules, update the docs

The refactor is only finished when the old import paths are gone — otherwise the next contributor adds code against them.

**Files:**
- Delete: `app/errors.py`, `app/imaging.py`, `app/matching/`, `app/store/`, `app/db/`, `app/jobs/`, `app/pipeline/`, `app/infrastructure/pipeline/image_quality.py`
- Modify: `app/infrastructure/pipeline/metrics.py` (keeps `_crop_face_bgr`, which is why it survives), every other importer of a deleted path (tests included)
- Create: `tests/test_layering.py` (replacing `tests/test_domain_layering.py`)
- Modify: `README.md`, `docs/` endpoint list if the tag names appear there

**Interfaces:**
- Consumes: everything.
- Produces: no new runtime interface. Produces the enforced invariant that `app/domain/` and `app/application/` cannot reach outward.

- [ ] **Step 1: Find every importer of a shimmed path**

```bash
graphify query "which modules import app.errors, app.imaging, app.store, app.db, app.jobs, app.pipeline or app.matching"
grep -rn "from app\.\(errors\|imaging\|store\|db\|jobs\|pipeline\|matching\)" app tests scripts
```

Write the list into the task log before changing anything — it is the checklist for Step 2.

- [ ] **Step 2: Rewrite each import to its new home, then delete the shims**

Mapping:

| Old | New |
|---|---|
| `app.errors` | `app.domain.errors` |
| `app.imaging` | `app.infrastructure.imaging` |
| `app.matching.cosine` | `app.domain.matching.cosine` |
| `app.matching.grouping` / `.ann_grouping` | `app.domain.matching.grouping` / `.ann_grouping` |
| `app.store.base` | `app.domain.ports.store` |
| `app.store.memory` / `.qdrant` | `app.infrastructure.store.memory` / `.qdrant` |
| `app.db.*` | `app.infrastructure.db.*` |
| `app.jobs.group_rebuild` | `app.infrastructure.jobs.group_rebuild` |
| `app.pipeline.base` | `app.domain.face` (`DetectedFace`) / `app.domain.ports.pipeline` (`FacePipeline`) |
| `app.pipeline.planarity` | `app.domain.planarity` |
| `app.pipeline.image_quality` | `app.domain.checks.chains` + `app.domain.checks.base` |
| `app.pipeline.*` (rest) | `app.infrastructure.pipeline.*` |

Then:

```bash
git rm app/errors.py app/imaging.py
git rm -r app/matching app/store app/db app/jobs app/pipeline
```

Any test that imported `ensure_face_quality` should now build a `Frame` and run `build_quality_chain` — the Task 4 tests show the pattern. Delete the `ensure_face_quality` shim along with its module; do not leave a deprecated function behind.

- [ ] **Step 3: Replace the layering test with one covering both inner layers**

```bash
git mv tests/test_domain_layering.py tests/test_layering.py
```

Rewrite it to assert two rules instead of one:

```python
import ast
import pathlib

DOMAIN = pathlib.Path("app/domain")
APPLICATION = pathlib.Path("app/application")

DOMAIN_FORBIDDEN_PREFIXES = (
    "app.application", "app.infrastructure", "app.api", "app.config",
    "app.main", "app.bootstrap",
)
APPLICATION_FORBIDDEN_PREFIXES = (
    "app.infrastructure", "app.api", "app.config", "app.main", "app.bootstrap",
)
FORBIDDEN_TOP = (
    "fastapi", "pydantic", "pydantic_settings", "sqlalchemy", "cv2",
    "qdrant_client", "insightface", "onnxruntime",
)


def _imported_modules(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
    return names


def _offenders(root: pathlib.Path, prefixes: tuple[str, ...]) -> list[str]:
    found = []
    for path in root.rglob("*.py"):
        for module in _imported_modules(path):
            if module.startswith(prefixes) or module.split(".")[0] in FORBIDDEN_TOP:
                found.append(f"{path}: {module}")
    return sorted(found)


def test_domain_package_exists():
    assert (DOMAIN / "errors.py").is_file()
    assert (APPLICATION / "params.py").is_file()


def test_domain_imports_nothing_outward():
    assert _offenders(DOMAIN, DOMAIN_FORBIDDEN_PREFIXES) == []


def test_application_imports_only_domain():
    assert _offenders(APPLICATION, APPLICATION_FORBIDDEN_PREFIXES) == []


def test_old_import_paths_are_gone():
    for stale in ("app/errors.py", "app/imaging.py", "app/store", "app/db",
                  "app/jobs", "app/pipeline", "app/matching", "app/deps.py",
                  "app/api/routes.py"):
        assert not pathlib.Path(stale).exists(), f"{stale} should have been deleted"
```

- [ ] **Step 4: Run it to verify it fails, then fix what it names**

Run: `python -m pytest tests/test_layering.py -v`
Expected: `test_application_imports_only_domain` FAILS on `app/application/queries/groups.py` or similar if any application module still reaches into infrastructure. Fix by moving the offending import behind a port — never by widening `APPLICATION_FORBIDDEN_PREFIXES`.

One known offender to resolve: `StartGroupRebuildUseCase` receives `settings` as an untyped argument but does not import `app.config`, so it passes. If any other use case imports `Settings` for a type hint, delete the hint rather than the rule.

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest -q`
Expected: zero failures, and the count is the Task 0 baseline plus every test added in Tasks 0–11, minus any test deleted in Step 2 for importing a now-gone helper. Note the arithmetic in the commit message.

- [ ] **Step 6: Verify the four Swagger sections by eye**

```bash
python - <<'PY'
from collections import Counter
from app.main import create_app
spec = create_app(testing=True).openapi()
print("tag order:", [t["name"] for t in spec["tags"]])
counts = Counter(
    tag
    for methods in spec["paths"].values()
    for op in methods.values()
    for tag in op.get("tags", [])
)
print("counts:", dict(counts))
assert [t["name"] for t in spec["tags"]] == ["Faces", "Liveness", "Groups", "System"]
assert dict(counts) == {"Faces": 10, "Liveness": 1, "Groups": 4, "System": 2}
print("OK")
PY
```

Expected: `OK`.

- [ ] **Step 7: Smoke-test the live app**

```bash
uvicorn app.main:app --port 8099 &
sleep 4
curl -s localhost:8099/health
curl -s localhost:8099/v1/faces/count
curl -s localhost:8099/v1/stats
kill %1
```

Expected: `/health` returns `{"status":"ok"}` with models and Qdrant up, or `503 {"detail":"dependencies not ready"}` without them — either is acceptable, a 500 or a traceback is not. If the lifespan raises, read the `RuntimeError` message: it names which of the three dependencies failed.

- [ ] **Step 8: Refresh the knowledge graph and the docs**

```bash
graphify update .
```

Update `README.md`: the module layout section, and the endpoint list if it groups endpoints differently from the four Swagger tags. Add a short "Architecture" paragraph naming the four layers and the dependency rule, and link the spec:

```markdown
## Architecture

Four layers, dependencies pointing inward only:

- `app/domain/` — errors, value objects, `Protocol` ports, pure check strategies. Imports nothing from the other layers.
- `app/application/` — 9 use cases and 3 query services over the ports.
- `app/infrastructure/` — InsightFace, ONNX, Qdrant, SQLAlchemy, OpenCV adapters.
- `app/api/` — 4 routers (Faces, Liveness, Groups, System), schemas, dependency providers, the error→status map.
- `app/bootstrap/` — composition root. `app/config.py` is read here and in `app/api/params_builder.py` only.

`tests/test_layering.py` fails the build if an inner layer reaches outward.
See [the design spec](docs/superpowers/specs/2026-10-28-clean-architecture-solid-design.md).
```

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "refactor: delete compatibility shims and enforce the layer rules

Every old import path is gone; app/domain and app/application are now
guarded by an AST test that fails on an outward import. README documents
the four layers and the dependency rule.

Suite: <N> passed (baseline <B> + <N-B> new)."
```

- [ ] **Step 10: Final diff review**

```bash
git log --oneline main..HEAD
git diff main...HEAD --stat
```

Expected: 12 commits, `app/api/routes.py` shows `-541`, `app/main.py` shows roughly `-128`. Read `git diff main...HEAD -- tests/` in full: every behavioural assertion that existed before must still exist somewhere.

---

## Notes for the executor

- **The suite is red only between Task 9 and Task 10.** Every other task ends green. If you are running with per-task review gates, treat 9 and 10 as one unit.
- **When a legacy error string and this plan disagree, the legacy string wins.** Copy it from the source file; the plan may have a typo, the running service does not.
- **Do not change the OpenAPI baseline.** If `tests/test_openapi_contract.py` fails, a response shape changed. That is the bug.
- **graphify first.** `graphify query "<question>"` before grepping; `graphify update .` after each task's code change.
