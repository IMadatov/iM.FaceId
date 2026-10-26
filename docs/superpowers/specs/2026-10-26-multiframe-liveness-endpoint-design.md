# iM.FaceId — Multi-frame Passive Liveness Endpoint (Design)

**Date:** 2026-10-26  
**Status:** Approved (pending implementation plan)  
**Project:** iM.FaceId  
**Parent:** [2026-10-05-im-faceid-webapi-design.md](./2026-10-05-im-faceid-webapi-design.md)  
**Related:** [2026-10-23-face-check-endpoint-design.md](./2026-10-23-face-check-endpoint-design.md), [2026-10-14-face-image-quality-gate-design.md](./2026-10-14-face-image-quality-gate-design.md), [2026-10-16-client-face-param-overrides-design.md](./2026-10-16-client-face-param-overrides-design.md)

## 1. Intent

Strengthen passive liveness against **photo / screen (static) spoof attacks** by accepting **multiple frames** in one request and combining two signals: the aggregated per-frame anti-spoof score **and** the inter-frame variation between the frames' face embeddings. A static photo held to the camera produces near-identical frames (variation ≈ 0); a real face produces natural micro-variation.

This ships as a **new, standalone endpoint** `POST /v1/faces/liveness`. The existing single-image `POST /v1/faces/check` is **not changed**.

**Success looks like:** a client captures N frames (N ≥ 2) and calls `POST /v1/faces/liveness`; a 200 means the frames passed both the mean-score gate and the inter-frame-distance gate; a held-still photo or screen replay (near-identical frames) is rejected with `422`; no face is stored and no embedding is returned.

## 2. Goals and success criteria

| Goal | Success |
|------|---------|
| Multi-frame request | Body is new `FaceLivenessRequest` with `images_base64: list[str]` (min 2) |
| Aggregated score | Decision uses the **mean** of per-frame liveness scores vs `liveness_threshold` |
| Anti-static-photo signal | Decision also requires **max pairwise cosine distance** ≥ `liveness_min_distance` |
| Always-on liveness | No `liveness` flag; quality gate + score + distance always run |
| No store side effects | `store` is not injected / called; vector DB unchanged |
| No embedding in response | Response has no vector / `face_id` |
| Clear signal | `200` + `{ ok, liveness_score, frame_count, inter_frame_distance }` |
| Docs + tests | `CLIENT_API.md` + README updated; API tests cover pass / too-similar / low-score / validation |

**Non-goals:** active challenge-response (blink/turn/smile); defeating high-fidelity video replay or 3D masks; changing `/v1/faces/check` or enroll/update/search/verify; returning embeddings or bbox; new depth/optical-flow model.

## 3. Constraints (agreed)

| Decision | Choice |
|----------|--------|
| Attack targeted | Photo / screen (static), passive — no user action |
| Mechanism | Per-frame passive model score (mean) **+** inter-frame embedding comparison |
| Variation signal | Cosine distance between per-frame InsightFace embeddings (reuse `detect_bgr` output; no new deps) |
| API surface | New standalone endpoint; existing `/v1/faces/check` untouched |
| Path | `POST /v1/faces/liveness` |
| Liveness flag | Removed — endpoint is always-on liveness |
| Min frames | 2 (distance signal needs ≥ 2 frames to detect a static photo) |
| Response | `ok` + `liveness_score` (mean) + `frame_count` + `inter_frame_distance`; no vector |
| Errors | Reuse existing error types / status codes |

## 4. Architecture

```
Client
  │  POST /v1/faces/liveness  (FaceLivenessRequest: images_base64[2..N])
  ▼
routes.check_liveness
  │  checker.ready()?                         ← else DependencyUnavailableError
  │  for each frame:
  │     _decode_base64_image                  ← InvalidImageError (msg includes frame index)
  │     pipeline.detect_bgr                    ← exactly 1 face, else FaceDetectionError (frame index)
  │     ensure_face_quality(bbox, thresholds)  ← existing quality gate, per frame
  │     score = checker.score_bgr(img, bbox)   ← per-frame passive score (no per-frame reject)
  │  mean_score = mean(scores)
  │  max_dist   = max( 1 - cosine_similarity(emb_i, emb_j) ) for i < j
  │  mean_score < liveness_threshold     → LivenessFailedError("mean score ...")
  │  max_dist   < liveness_min_distance  → LivenessFailedError("frames too similar ...")
  │  (no store.upsert)
  ▼
LivenessResponse { ok: true, liveness_score: mean_score, frame_count, inter_frame_distance: max_dist }
```

Dependencies: `settings`, `pipeline`, `liveness` checker. **Do not inject / call `store`.**

Notes:
- Per-frame scoring uses `checker.score_bgr(img, bbox)` (not `ensure_live`) so a single weak frame does not short-circuit; the mean is what the threshold is applied to.
- Embeddings come from `pipeline.detect_bgr(...).embedding`; InsightFace embeddings are L2-normalized (the fake pipeline normalizes too), so cosine distance is `1 - cosine_similarity`.
- `detect_bgr` computes an embedding internally today; it is used only for the distance signal, never returned or persisted.

## 5. API contract

### Request — `FaceLivenessRequest` (new)

```json
{
  "images_base64": ["<base64>", "<base64>"],
  "blur_min_variance": null,
  "brightness_min": null,
  "brightness_max": null,
  "min_side_px": null,
  "liveness_threshold": null,
  "liveness_min_distance": null
}
```

- `images_base64`: required list, `min_length = 2`, `max_length = liveness_max_frames`.
- Tuning fields optional; `None` → fall back to `Settings` defaults via existing `_resolve`.
- `model_config = ConfigDict(extra="forbid")`.

### Response 200 — `LivenessResponse` (new)

```json
{
  "ok": true,
  "liveness_score": 0.97,
  "frame_count": 3,
  "inter_frame_distance": 0.11
}
```

All four fields always populated (endpoint is always-on).

### Errors

Reuse existing types and their current status mapping:

| Condition | Error |
|-----------|-------|
| Fewer than 2 or more than `liveness_max_frames` frames | pydantic `422` (validation) |
| Bad / oversized frame | `InvalidImageError` (message includes frame index) |
| No / multiple faces in a frame | `FaceDetectionError` (message includes frame index) |
| Frame fails blur/brightness/size | `ImageQualityError` |
| Mean score below threshold | `LivenessFailedError` ("mean score ...") |
| Frames too similar (static photo suspected) | `LivenessFailedError` ("frames too similar ...") |
| Liveness model not installed | `DependencyUnavailableError` |

No new error types.

## 6. Configuration (new `Settings` fields)

| Setting | Default | Notes |
|---------|---------|-------|
| `liveness_max_frames` | `5` | Upper bound on `images_base64` length |
| `liveness_min_distance` | `0.05` | **NEEDS TUNING** — min acceptable max-pairwise cosine distance. Calibrate on real captures: a static photo yields ≈ 0; a real face's micro-variation is larger. Start conservative, tighten with data. |

`liveness_threshold` already exists and is reused for the mean-score gate.

## 7. Implementation sketch

| File | Change |
|------|--------|
| `app/api/schemas.py` | Add `FaceLivenessRequest` and `LivenessResponse` |
| `app/api/routes.py` | Add `check_liveness` handler + `_run_multiframe_liveness` helper (decode/detect/quality/score per frame, aggregate, distance) |
| `app/config.py` | Add `liveness_max_frames`, `liveness_min_distance` |
| `app/matching/cosine.py` | Add `cosine_distance(a, b) = 1 - cosine_similarity(a, b)` (or compute inline) |
| `docs/CLIENT_API.md`, `README.md` | Document `POST /v1/faces/liveness` |
| `tests/test_api_faces.py` | Cases in §8 |

Route registration order: register `/v1/faces/liveness` **before** any `/v1/faces/{face_id}` route so `"liveness"` is not captured as a path param (same pattern as existing `/v1/faces/check`, `/v1/faces/search`, `/v1/faces/list`).

## 8. Testing (TDD)

The fake pipeline derives each embedding deterministically from image content, so identical frames → distance ≈ 0 and distinct frames → distance > 0 (no test-double changes needed).

- **Happy path:** 2–3 distinct valid face frames, high fake score → `200`, `ok: true`, `inter_frame_distance > liveness_min_distance`, `frame_count` correct, store count unchanged.
- **Static photo:** identical frames repeated → `422` `LivenessFailedError` ("too similar").
- **Low score:** `FakeLivenessChecker(force_spoof=True)` → `422` (mean below threshold).
- **Too few frames:** 1 frame → `422` validation.
- **Too many frames:** `liveness_max_frames + 1` → `422` validation.
- **Bad frame:** one frame with no face → `FaceDetectionError` status, message names the frame index.
- **Model unavailable:** unavailable checker → `DependencyUnavailableError` status.
- **No store write:** `GET /v1/faces/count` equals pre-call value.

## 9. Out of scope

- Active challenge-response liveness.
- Depth / optical-flow / moiré-specific models.
- Applying multi-frame to enroll / verify / search / update / check.
- Returning embeddings, bbox, or per-frame score breakdown (only the aggregate + max distance are returned).
- Auth / rate-limit changes specific to this route.
