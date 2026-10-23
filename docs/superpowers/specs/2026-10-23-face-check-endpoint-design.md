# iM.FaceId — Face Check Endpoint (Design)

**Date:** 2026-10-23  
**Status:** Approved (pending implementation plan)  
**Project:** iM.FaceId  
**Parent:** [2026-10-05-im-faceid-webapi-design.md](./2026-10-05-im-faceid-webapi-design.md)  
**Related:** [2026-10-14-face-image-quality-gate-design.md](./2026-10-14-face-image-quality-gate-design.md), [2026-10-16-client-face-param-overrides-design.md](./2026-10-16-client-face-param-overrides-design.md)

## 1. Intent

Add a **preflight** HTTP endpoint that accepts the same image + optional quality/liveness body as enroll, runs the same decode → detect → quality → liveness path, and tells the client whether the image is acceptable — **without writing to the vector store** and **without returning an embedding**.

**Success looks like:** clients can call `POST /v1/faces/check` before enroll; a 200 means the same checks enroll would pass; failures use the same error shapes as enroll; no face is stored.

## 2. Goals and success criteria

| Goal | Success |
|------|---------|
| Same request as enroll | Body is `FaceImageRequest` (reuse schema) |
| Same gates as enroll | Decode, `detect_bgr`, quality, optional liveness via existing helpers |
| No store side effects | `store` is not called; vector DB unchanged |
| No embedding in response | Response has no vector / `face_id` |
| Clear success signal | `200` + `{ "ok": true, "liveness_score": ... }` |
| Docs + tests | `CLIENT_API.md` updated; API tests cover pass / quality fail / no store write |

**Non-goals:** detect-only pipeline without recognition model; dry-run flag on enroll; returning embedding or bbox; changing enroll/update/search/verify behavior.

## 3. Constraints (agreed)

| Decision | Choice |
|----------|--------|
| Approach | Enroll clone without `store.upsert` |
| Path | `POST /v1/faces/check` |
| Purpose | Image quality / readiness check for clients |
| Response | `ok` + optional `liveness_score`; no vector |
| Errors | Same as enroll (status codes and payloads) |

## 4. Architecture

```
Client
  │  POST /v1/faces/check  (FaceImageRequest)
  ▼
routes.check_face
  │  _decode_base64_image
  │  pipeline.detect_bgr          ← same as enroll (embedding computed internally, discarded)
  │  _run_quality_and_liveness    ← same helpers / thresholds
  │  (no store.upsert)
  ▼
CheckResponse { ok: true, liveness_score }
```

Dependencies: `settings`, `pipeline`, `liveness` checker. **Do not inject / call `store`.**

Note: InsightFace `detect_bgr` still produces an embedding internally today. That is acceptable; it is neither returned nor persisted.

## 5. API contract

### Request

Same as enroll / update:

```json
{
  "image_base64": "<base64>",
  "liveness": false,
  "blur_min_variance": null,
  "brightness_min": null,
  "brightness_max": null,
  "min_side_px": null,
  "liveness_threshold": null
}
```

### Response 200

```json
{
  "ok": true,
  "liveness_score": null
}
```

`liveness_score` is set only when `liveness` was `true` (same rule as enroll).

### Errors

Identical to enroll for the shared prefix of the pipeline (invalid/oversized image, no/multiple face as enroll defines them, `ImageQualityError`, `LivenessFailedError`, dependency unavailable). No new error types.

## 6. Implementation sketch

| File | Change |
|------|--------|
| `app/api/schemas.py` | Add `CheckResponse(ok: bool = True, liveness_score: float \| None = None)` |
| `app/api/routes.py` | Add `check_face` handler; mirror `enroll_face` without `uuid` / `store.upsert` |
| `docs/CLIENT_API.md` | Document `POST /v1/faces/check` |
| `tests/test_api_faces.py` | Pass case; quality/liveness fail cases; assert store count unchanged |

Route registration order: register `/v1/faces/check` **before** any `/v1/faces/{face_id}` routes so `"check"` is not captured as a path param (same pattern as existing `/v1/faces/search`, `/v1/faces/list`).

## 7. Testing

- Happy path: valid face image → 200, `ok: true`, store count unchanged.
- With `liveness: true` (fake checker): 200 and non-null `liveness_score` when applicable.
- Quality fail / detect fail: same status as enroll for the same fixture.
- Explicit regression: after check, `GET /v1/faces/count` (or store fixture count) equals pre-call value.

## 8. Out of scope

- Returning embedding, bbox, or quality metrics breakdown.
- Skipping embedding compute inside the pipeline.
- Auth / rate-limit changes specific to this route.
