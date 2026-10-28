# iM.FaceId — Clean Architecture + SOLID Restructure (Design)

**Date:** 2026-10-28
**Status:** Approved (pending implementation plan)
**Project:** iM.FaceId
**Parent:** [2026-10-05-im-faceid-webapi-design.md](./2026-10-05-im-faceid-webapi-design.md)
**Related:** [2026-10-26-multiframe-liveness-endpoint-design.md](./2026-10-26-multiframe-liveness-endpoint-design.md), [2026-10-12-ann-graph-grouping-design.md](./2026-10-12-ann-graph-grouping-design.md), [2026-10-14-face-image-quality-gate-design.md](./2026-10-14-face-image-quality-gate-design.md)

## 1. Intent

Restructure the service into four layers with an inward-only dependency rule, and bring the code
in line with SOLID so that adding capability does not mean editing existing code. The driver is
growth: the team expects to keep adding liveness checks, storage backends, and endpoints.

Today all 17 endpoints and their business logic live in a single 541-line
`app/api/routes.py`. Orchestration sits in private helpers next to HTTP handlers
(`_run_quality_and_liveness`, `_run_multiframe_liveness` — a 74-line algorithm, `_check_planarity`),
so none of it can be tested or reused without a `TestClient`. There is no application layer; routers
reach straight into `store`, `pipeline`, and `GroupsRepository`, and three handlers open and close
SQLAlchemy sessions by hand.

**This is a behaviour-preserving refactor.** No endpoint path, request body, response shape, status
code, or `detail` string changes. The only intentional change to the OpenAPI document is the
addition of four `tags` sections.

**Success looks like:** the full test suite passes unchanged except for import paths; an OpenAPI
snapshot test proves the contract is byte-identical apart from `tags`; adding a new liveness check
means adding one file and one factory line, with no existing file edited; a use case can be tested
with a one-method fake instead of a nine-method one.

## 2. Goals and success criteria

| Goal | Success |
|------|---------|
| Four layers, inward-only imports | `domain/` imports nothing from the project; `application/` imports only `domain/`; only `api/` and `bootstrap/` know every layer |
| Controllers split by resource | `app/api/routers/{faces,liveness,groups,system}.py`; `routes.py` deleted |
| Swagger sections | Four tags — `Faces` (10), `Liveness` (1), `Groups` (4), `System` (2) — ordered and described via `openapi_tags` |
| Business logic out of HTTP | 9 use-case classes + 3 query services; no handler contains orchestration |
| ISP on the store | `FaceStore` (9 methods) split into 5 role interfaces; each use case depends only on what it calls |
| DIP complete | Protocol ports for liveness, groups, imaging, id generation, rebuild lock, unit of work |
| OCP on checks | Each quality/liveness check is its own class behind a `FrameCheck` or `SequenceCheck` protocol, composed in an ordered chain |
| No HTTP in logic | The 5 inline `HTTPException` raises become domain exceptions mapped in one place |
| Typed composition root | `app.state.container` replaces 5 stringly-typed `getattr(app.state, ...)` lookups |
| Behaviour unchanged | OpenAPI snapshot test passes; existing tests pass with only import-path edits |
| Extracted logic covered | New HTTP-free unit tests for every check, use case, and query service |

**Non-goals:** changing any endpoint contract; adding endpoints; tuning thresholds or liveness
accuracy; a distributed rebuild lock (see §10); replacing Qdrant, Postgres, or InsightFace;
async/await conversion; authentication.

## 3. Constraints (agreed)

| Decision | Choice |
|----------|--------|
| Layering | Four layers, pragmatic — no DTO mapping ceremony at every boundary |
| Application layer shape | Use-case classes for endpoints with policy; grouped query services for store passthroughs |
| Scope | Pure refactor **plus** new unit tests for extracted logic; no behaviour changes, no opportunistic fixes |
| Swagger | Four tags matching the router split |
| Prerequisite | `feat/face-device-auto-gpu` merges into `main` **first** (see §4) |
| Rebuild lock | Extract the port now; keep the existing `threading.Lock` adapter; distributed adapter deferred |
| `cv2`/`numpy` in domain | Allowed — the domain of this service *is* image mathematics |
| `Settings` | Stays at `app/config.py`; does **not** enter `application/` or `domain/` |

## 4. Prerequisite: merge the GPU branch first

`feat/face-device-auto-gpu` is unmerged and 8 commits ahead. It changes `app/main.py` (70 lines),
`app/pipeline/onnx_insightface.py`, `app/pipeline/onnx_liveness.py`, and adds
`app/pipeline/device.py` plus 3 test files — all inside the blast radius of this refactor.

Refactoring first would make that branch a near-manual re-apply, because `main.py` is dissolved into
`bootstrap/` and `pipeline/` moves under `infrastructure/`. So: **merge it, confirm the suite is
green, snapshot the OpenAPI document, then start.** The baseline test count is 162 on `main` today
and rises with that branch's new files.

## 5. The dependency rule

```
api ──→ application ──→ domain
 │                        ▲
 └──→ infrastructure ─────┘
```

- `domain/` — imports nothing from `app.*`. No FastAPI, pydantic, SQLAlchemy, or qdrant-client.
  `numpy` and `cv2` are permitted: the quality and planarity checks are pure functions over image
  arrays, with no I/O or external system.
- `application/` — imports only `domain/`. Knows neither FastAPI nor `Settings`.
- `infrastructure/` — implements `domain/ports/`. Owns every library that talks to the outside world.
- `api/` and `bootstrap/` — the only places allowed to see all layers, and the only places that wire
  them together.

### Why `Settings` stays out of `application/`

`_resolve(client_value, default)` — merging a client override with a settings default — is called
**12 times** across `routes.py`, 7 of them inside `_run_multiframe_liveness` alone. That resolution
is an API-layer concern. It happens once per request in the router, which hands the application layer
a finished value object:

```python
# app/application/params.py
@dataclass(frozen=True)
class GateParams:
    blur_min_variance: float
    brightness_min: float
    brightness_max: float
    min_side_px: float
    liveness_threshold: float

@dataclass(frozen=True)
class MultiframeParams:
    gate: GateParams
    max_frames: int
    min_distance: float
    min_planarity: float
    min_view_change: float
```

Use cases then have no dependency on `.env`, `pydantic-settings`, or defaults — a test constructs
the dataclass directly.

## 6. SOLID application

### 6.1 ISP — split the fat store interface

`app/store/base.py` declares `FaceStore` with **9 methods**. A new backend must implement all 9, and
`verify_face`, which only needs `get`, is coupled to `clear`, `list_all`, and `ping`.

```python
# app/domain/ports/store.py
class FaceReader(Protocol):
    def get(self, face_id: str) -> np.ndarray | None: ...

class FaceWriter(Protocol):
    def upsert(self, face_id: str, vector: np.ndarray) -> None: ...
    def delete(self, face_id: str) -> bool: ...

class FaceSearcher(Protocol):
    def search(
        self, vector: np.ndarray, *, limit: int, min_score: float
    ) -> list[tuple[str, float]]: ...

class FaceCatalog(Protocol):
    def count(self) -> int: ...
    def list_ids(self, *, limit: int, offset: int) -> list[str]: ...
    def list_all(self) -> list[tuple[str, np.ndarray]]: ...
    def clear(self) -> int: ...

class HealthProbe(Protocol):
    def ping(self) -> bool: ...
```

Because `typing.Protocol` is structural, `QdrantFaceStore` and `MemoryFaceStore` satisfy all five
**without any declaration change** — the cost to adapters is zero. The gain is at the consumer:
`EnrollFace(writer=...)`, `VerifyFace(reader=...)`, `SearchFaces(searcher=...)` state exactly what
they touch, and their test fakes shrink from nine methods to one.

### 6.2 OCP — the check chain

This is the highest-value change for the stated growth plan, because quality and liveness checks are
precisely what grows in a FaceID service. Today adding one means **editing**
`_run_quality_and_liveness` and `_run_multiframe_liveness`.

Two protocols are needed, not one, because the checks divide into two genuinely different shapes:
some judge a single frame, others judge a sequence. Forcing both behind one signature would mean a
`MeanLivenessCheck` that ignores its arguments — an LSP violation written into the interface.

```python
# app/domain/checks/base.py
@dataclass(frozen=True)
class Frame:
    image_bgr: np.ndarray
    detected: DetectedFace

@dataclass(frozen=True)
class FrameSequence:
    frames: list[Frame]
    liveness_scores: list[float]

class FrameCheck(Protocol):
    name: str
    def verify(self, frame: Frame) -> None: ...        # raises on failure

class SequenceCheck(Protocol):
    name: str
    def verify(self, sequence: FrameSequence) -> None: ...
```

One class per check, each holding its own threshold, constructed by a factory:

| Check class | Protocol | Replaces |
|---|---|---|
| `BlurCheck`, `BrightnessCheck`, `MinSizeCheck` | `FrameCheck` | `ensure_face_quality` internals |
| `LivenessScoreCheck` | `FrameCheck` | `_maybe_liveness` |
| `MeanLivenessCheck` | `SequenceCheck` | mean-score gate in `_run_multiframe_liveness` |
| `InterFrameDistanceCheck` | `SequenceCheck` | distance gate in `_run_multiframe_liveness` |
| `ViewChangeCheck`, `PlanarityCheck` | `SequenceCheck` | `_check_planarity` |

Composed by two chains that run members in order and raise on the first failure:

- single-frame gate — `FrameCheckChain([Blur, Brightness, MinSize, LivenessScore])`
- multi-frame — `FrameCheckChain([Blur, Brightness, MinSize])` applied per frame, then
  `SequenceCheckChain([MeanLiveness, InterFrameDistance, ViewChange, Planarity])`

The single-frame gate and the multi-frame per-frame gate differ by exactly one member
(`LivenessScoreCheck`), which matches today's behaviour: `_run_quality_and_liveness` enforces a
per-image score, while `_run_multiframe_liveness` collects per-frame scores and judges only their
mean.

Adding a check at scale becomes: one new file, one factory line. No existing file is edited.

**Hard requirement:** evaluation order and failure messages are part of the observable contract and
tests assert on them. The multi-frame order must stay: per-frame quality → mean score → inter-frame
distance → view change → planarity. Every message string is preserved verbatim, including the
`frame[{index}]:` prefixes applied to decode and detection failures.

**Preserved edge case:** `_check_planarity` returns `None` — and both motion checks are skipped —
when `min_planarity <= 0` and `min_view_change <= 0`. `ViewChangeCheck` and `PlanarityCheck` are
therefore omitted from the chain entirely in that configuration, so `planarity_residual` stays
`None` in the response. When keypoints are missing but the checks are active, the existing
`DependencyUnavailableError` with its current message is raised.

### 6.3 DIP — close the remaining holes

| Hole | Today | After |
|---|---|---|
| Liveness checker | **No protocol at all** — duck-typed off `app.state.liveness` | `domain/ports/liveness.py` |
| Groups repository | Concrete `GroupsRepository` imported by handlers | `domain/ports/groups.py` |
| Id generation | `uuid.uuid4()` called inline at `routes.py:159` | `IdGenerator` port — makes enroll deterministic under test |
| Image decoding | `decode_and_resize` imported directly | `ImageDecoder` port |
| Session lifecycle | `session = session_factory()` / `.close()` in 3 handlers | `UnitOfWork` port |
| Rebuild single-flight | `threading.Lock` owned by the service | `RebuildLock` port |
| App wiring | `getattr(request.app.state, "store", None)` × 5, untyped | typed `Container` dataclass at `app.state.container` |

The recovered `LivenessChecker` contract, read off the existing implementations:

```python
class LivenessChecker(Protocol):
    def ready(self) -> bool: ...
    def score_bgr(self, image_bgr: np.ndarray, *, bbox: np.ndarray | None) -> float: ...
    def ensure_live(
        self, image_bgr: np.ndarray, *, threshold: float, bbox: np.ndarray | None = None
    ) -> float: ...
```

The typed container replaces five independent string-keyed lookups with one failure mode that a type
checker can see.

### 6.4 SRP — and one deliberate deviation

A purist reading gives 17 use-case classes, one per endpoint. Five of those endpoints — `count`,
`clear`, `get` (exists), `delete`, `list` — are two-line passthroughs to the store with no policy.
They answer to one actor (ops) and change for one reason. SRP is "one reason to change," not "one
method per class," so grouping them is not a violation; splitting them would add files without
adding isolation.

```
app/application/
  use_cases/                        # policy — each will grow independently
    enroll_face.py                  EnrollFace
    check_face.py                   CheckFace
    verify_face.py                  VerifyFace
    search_faces.py                 SearchFaces
    update_face.py                  UpdateFace
    check_multiframe_liveness.py    CheckMultiframeLiveness
    start_group_rebuild.py          StartGroupRebuild
    execute_group_rebuild.py        ExecuteGroupRebuild
    list_face_groups.py             ListFaceGroups
  queries/                          # no policy — passthrough, single actor
    face_catalog.py                 count / clear / exists / delete / list
    group_runs.py                   status / list_runs
    system_status.py                health / stats
  gate.py                           FaceGate — shared by 5 use cases
  params.py                         GateParams · MultiframeParams
```

Use cases return plain dataclasses, never pydantic models; the router maps result → response. This
keeps pydantic out of `application/` and lets every use case be driven without HTTP.

### 6.5 SRP — `GroupRebuildService` is a clear violation

`app/jobs/group_rebuild.py` currently does four jobs in one class: acquiring and releasing a
`threading.Lock`, opening and closing sessions (including a second session purely for error
persistence), orchestrating the rebuild, and persisting failures. Four reasons to change. It splits:

```
application/use_cases/start_group_rebuild.py     orchestration
application/use_cases/execute_group_rebuild.py   orchestration
infrastructure/db/unit_of_work.py                session lifecycle
domain/ports/rebuild_lock.py                     single-flight port
infrastructure/locks/process_lock.py             threading.Lock adapter
```

### 6.6 LSP

The fakes (`FakeFacePipeline`, `FakeLivenessChecker`, `UnavailableLivenessChecker`,
`MemoryFaceStore`) already stand in for the real adapters in 162 tests, so substitutability holds
today. Writing the ports down makes it checkable rather than incidental: each fake must satisfy the
same protocol as its production counterpart, and `UnavailableLivenessChecker` must honour
`LivenessChecker` by returning `False` from `ready()` rather than raising.

## 7. Error handling

`routes.py` raises `HTTPException` in five places **inside business logic**, so application-level
decisions currently depend on HTTP. Each becomes a domain exception, mapped in exactly one place.

| Today | Location | Becomes | Status | `detail` |
|---|---|---|---|---|
| `HTTPException(503, ...)` | `routes.py:142` | `DependencyUnavailableError` | 503 | `dependencies not ready` |
| `HTTPException(422, ...)` | `routes.py:187` | `TooManyFramesError` | 422 | `too many frames (max {n})` |
| `GroupRebuildConflict` → 409 | `routes.py:363` | `RebuildInProgressError` | 409 | `group rebuild already running` |
| `ValueError` → 400 | `routes.py:365` | `InvalidParameterError` | 400 | `min_size must be >= 2` / `top_k must be >= 1` |
| `HTTPException(404, ...)` | `routes.py:412` | `GroupRunNotFoundError` | 404 | `group run {run_id} not found` |
| `HTTPException(400, ...)` | `routes.py:426` | `InvalidParameterError` | 400 | `min_size must be >= 2` |

Existing mappings are unchanged: `InvalidImageError` 400, `FaceNotFoundError` 404,
`FaceDetectionError` 422, `ImageQualityError` 422, `LivenessFailedError` 422,
`DependencyUnavailableError` 503.

Status codes and `detail` strings are preserved byte-for-byte. The new handlers keep the existing
response body shape, `{"detail": str(exc)}`, which is what FastAPI's own `HTTPException` handler
produces — so the five converted sites are indistinguishable to a client.

The single mapping table lives in `app/api/exception_handlers.py` as an explicit dict, replacing the
`_status: int = status` default-argument closure workaround at `main.py:36`.

## 8. Data flow

`POST /v1/faces` as the representative path:

```
api/routers/faces.py
  body: FaceImageRequest (pydantic)
  settings + body → GateParams                 ← _resolve happens here, once
  use_case = Depends(enroll_face_dep)          ← built from app.state.container
      │
application/use_cases/enroll_face.py
  EnrollFace.execute(image_b64, params, liveness_enabled) -> EnrollResult
      ├─ self._decoder.decode(image_b64)       ← ImageDecoder port
      ├─ self._pipeline.detect_bgr(img)        ← FacePipeline port
      ├─ self._gate.apply(Frame(img, detected))← FaceGate → FrameCheckChain
      ├─ face_id = self._ids.new()             ← IdGenerator port
      └─ self._writer.upsert(face_id, emb)     ← FaceWriter port
      │
api/routers/faces.py
  EnrollResult → FaceIdResponse (pydantic)
```

## 9. Component inventory

| Today | After | Note |
|---|---|---|
| `app/errors.py` | `app/domain/errors.py` | unchanged content; 7 test imports updated; 4 new exceptions added (§7) |
| `app/pipeline/base.py` | `app/domain/face.py` + `app/domain/ports/pipeline.py` | `DetectedFace` / `FacePipeline` |
| `app/store/base.py` | `app/domain/ports/store.py` | split into 5 role interfaces |
| `app/pipeline/image_quality.py` | `app/domain/checks/quality.py` | decomposed into `BlurCheck`, `BrightnessCheck`, `MinSizeCheck` |
| `app/pipeline/planarity.py` | `app/domain/planarity.py` + `app/domain/checks/motion.py` | math stays pure; gating becomes checks |
| `app/matching/cosine.py` | `app/domain/matching.py` | |
| `app/matching/{grouping,ann_grouping}.py` | `app/domain/grouping.py` | merged (71 + 50 lines) |
| `app/imaging.py` + `routes.py:_decode_base64_image` | `app/infrastructure/imaging.py` | base64 strip and decode together; explicit params, not `Settings` |
| `app/pipeline/{onnx_*,device,fake,fake_liveness}.py` | `app/infrastructure/pipeline/` | adapters |
| `app/store/{qdrant,memory}.py` | `app/infrastructure/store/` | adapters |
| `app/db/*` | `app/infrastructure/db/` | plus new `unit_of_work.py` |
| `app/jobs/group_rebuild.py` | split per §6.5 | directory removed |
| `routes.py` helpers | `app/application/` | per §6.4 |
| `app/api/routes.py` (541 lines) | `app/api/routers/` × 4 | file removed |
| `app/api/schemas.py` (172 lines) | `app/api/schemas/` × 4 | 25 models across `face`/`liveness`/`group`/`system` |
| `app/deps.py` | `app/api/dependencies.py` | composition root; typed `Container` |
| `main.py:_register_exception_handlers` | `app/api/exception_handlers.py` | |
| `main.py:_production_lifespan` + `_wire_database` | `app/bootstrap/lifespan.py` | |
| `main.py:create_app` | `app/bootstrap/app_factory.py` | plus `openapi_tags` |
| `app/main.py` | `app/main.py` | reduced to an import and `app = create_app()` |
| `app/config.py` | unchanged | cross-cutting; its 2 test imports are untouched |

### Route registration order — load-bearing

`GET /v1/faces/count` must be registered **before** `GET /v1/faces/{face_id}`, or the path parameter
swallows it and `count` returns 404. Today `routes.py` happens to satisfy this (line 310 vs line
502). Both must therefore stay in `faces.py` in that relative order. Every other pair differs in
segment count or method and cannot collide:

- `POST /v1/faces/{check,list,search,liveness}` (3 segments) vs `POST /v1/faces/{face_id}/verify` (4)
- `GET /v1/faces/groups/rebuild` (4) vs `GET /v1/faces/{face_id}` (3)
- `DELETE /v1/faces` (2) vs `DELETE /v1/faces/{face_id}` (3)

### Swagger tags

| Tag | Count | Endpoints |
|---|---|---|
| `Faces` | 10 | enroll, check, count, list, search, clear, get, update, delete, verify |
| `Liveness` | 1 | multi-frame liveness |
| `Groups` | 4 | rebuild (POST, GET), rebuild/{run_id}, groups/list |
| `System` | 2 | health, stats |

All 17 endpoints are currently untagged and render in one unnamed `default` block. `openapi_tags` on
the `FastAPI` constructor sets section order and descriptions; without it, order follows registration
and is effectively arbitrary.

## 10. Known limitation: multi-worker single-flight

Group rebuild must run one at a time. Both current guards fail across processes:

1. `group_rebuild.py:44` — `threading.Lock` lives in one process's memory.
2. `group_rebuild.py:50-52` — `repo.has_running()` then `create_run()` then `commit()` is a
   check-then-act race. `GroupRun` has **no unique constraint on `status`** (verified in
   `app/db/models.py`), so the database does not close the window either.

Under `uvicorn --workers N` or more than one replica, N rebuilds can start simultaneously.

This refactor does **not** fix it — that would be a behaviour change, outside the agreed scope, and
mixing it in would make any regression ambiguous. Extracting `RebuildLock` makes the fix cheap:
a `PostgresAdvisoryLock` adapter is one new file plus one line in `bootstrap/`, with no use case
touched. A `SELECT ... FOR UPDATE` or a partial unique index on `status` would close the second race.

## 11. Testing

### Existing tests are the behaviour contract

The 162 existing tests (plus those arriving with the GPU branch) are not rewritten. Only import
paths change. They are the only real safety net for a refactor of this size.

### OpenAPI snapshot test — added at step 0

```python
def test_openapi_contract_unchanged():
    """Proves the refactor changed no path, schema, or status code."""
    spec = create_app(testing=True).openapi()
    # compared against the committed pre-refactor snapshot, ignoring `tags`
```

Captured before any code moves, it answers "did the contract survive?" after every step.

### New unit tests

| Subject | Count | Rationale |
|---|---|---|
| Each check class | ~8 | reachable only through `TestClient` today |
| Chain ordering | 2 | order is observable; a reordering changes which error a client sees |
| Each use case | 9 | `CheckMultiframeLiveness` wraps a 74-line algorithm with no HTTP-free test today |
| Query services | 3 | |
| `GateParams` resolution | 2 | one site replacing 12 `_resolve` calls |
| `Container` wiring | 1 | asserts the typed container is fully populated |

Role-interface fakes keep these cheap: a `VerifyFace` test needs a one-method `FaceReader` fake, not
a nine-method `FaceStore`.

## 12. Migration order

Ordered so the **full suite passes after every step** — a regression is attributable to one commit.
Each row is one commit.

| # | Step | Expected state |
|---|---|---|
| 0 | Merge `feat/face-device-auto-gpu` into `main`; capture the OpenAPI snapshot | green (baseline) |
| 1 | Create `domain/`: errors, face, ports, matching, grouping, planarity — **moves only**; update imports | green |
| 2 | Create `infrastructure/`: imaging, pipeline, store, db, locks — **moves only**; update imports | green |
| 3 | Extract `domain/checks/` with both chains; `routes.py` delegates to them | green + ~10 new |
| 4a | `use_cases/`: enroll, check, update, verify, search, plus `FaceGate` | green + 5 new |
| 4b | `use_cases/check_multiframe_liveness.py` — the 74-line algorithm leaves HTTP | green + 1 new |
| 4c | `use_cases/{start,execute}_group_rebuild.py` + `UnitOfWork` + `RebuildLock`; `jobs/` removed | green + 2 new |
| 4d | `queries/`: face_catalog, group_runs, system_status | green + 3 new |
| 5 | Split `api/`: 4 routers (with tags), 4 schema modules, `dependencies.py` + typed `Container` | green; snapshot matches apart from `tags` |
| 6 | Split `bootstrap/`: lifespan, app_factory; `main.py` reduced | green |
| 7 | Final pass: full suite, OpenAPI diff, `graphify update .`, README and `docs/CLIENT_API.md` review | green |

Steps 1–2 and 5–6 are near-mechanical moves. The real risk is in 3 and 4b, where algorithms leave
their current home — hence their own commits.

## 13. Final structure

```
app/
  domain/
    errors.py  face.py  matching.py  grouping.py  planarity.py
    ports/     store · pipeline · liveness · groups · imaging · identity · rebuild_lock · unit_of_work
    checks/    base · quality · liveness · motion · chains
  application/
    use_cases/ (9)   queries/ (3)   gate.py   params.py
  infrastructure/
    imaging.py   pipeline/   store/   db/ (+ unit_of_work)   locks/
  api/
    routers/ (4)   schemas/ (4)   dependencies.py   exception_handlers.py
  bootstrap/
    lifespan.py   app_factory.py
  config.py
  main.py
```

Removed as paths: `app/api/routes.py`, `app/errors.py`, `app/imaging.py`, `app/deps.py`,
`app/jobs/`, `app/matching/`, `app/store/`, `app/pipeline/`, `app/db/`. Of these, `app/matching/`,
`app/store/`, `app/pipeline/`, and `app/db/` move under `domain/` or `infrastructure/`; `app/jobs/`
splits across `application/` and `infrastructure/` per §6.5.

## 14. Open questions

None. All design decisions are recorded in §3.
