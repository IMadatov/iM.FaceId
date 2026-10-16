# iM.FaceId — Client Face Parameter Overrides (Design)

**Date:** 2026-10-16  
**Status:** Approved  
**Project:** iM.FaceId  
**Parent:** [2026-10-05-im-faceid-webapi-design.md](./2026-10-05-im-faceid-webapi-design.md)  
**Related:** [2026-10-14-face-image-quality-gate-design.md](./2026-10-14-face-image-quality-gate-design.md)

## 1. Intent

Let API clients override face pipeline thresholds per request—image quality (blur, brightness, min face side), liveness score threshold, and verify match threshold—without changing server `.env` defaults.

**Success looks like:** omitting `options` preserves current behavior; sending a JSON `options` Form field changes effective thresholds for that call only; search continues to use existing Query `min_score` / `limit`.

## 2. Goals and success criteria

| Goal | Success |
|------|---------|
| Per-request overrides | Optional multipart Form field `options` (JSON object) on enroll, update, search, verify |
| Defaults | Omit `options` or omit a key → `Settings` / `.env` value |
| Full override | No server floors/ceilings; client may disable quality checks (same semantics as env: blur `0`, brightness `0`/`255`, min side `0`) |
| Scope of knobs | Quality + `liveness_threshold` on all liveness-capable image endpoints; `match_threshold` on verify; search score via existing Query `min_score` |
| Compatibility | Existing clients unchanged (`liveness` Form bool kept); search `min_score` stays Query |

## 3. Constraints (agreed)

| Decision | Choice |
|----------|--------|
| Transport | Multipart file + Form `liveness` + optional Form `options` (JSON string) |
| Override policy | Full override (option A) |
| Endpoints | All image endpoints that already take `liveness` (option A) |
| Search score | Keep Query `min_score` / `limit`; do not put in `options` |
| Quality when `liveness=false` | Overrides accepted but ignored (quality still only runs on liveness path) |
| Response shape | No new fields; verify/search already return effective `threshold` |
| Invalid JSON / unknown keys | HTTP `422` (`extra=forbid` on options model) |

## 4. Architecture

```
multipart: image + liveness + optional options (JSON string)
        │
        ▼
  parse FaceRequestOptions; resolve: key if set else settings.*
        │
        ▼
decode → detect_bgr
              │
              ├─ liveness=false → store / search / verify
              │     (quality + MiniFASNet skipped;
              │      verify uses resolved match_threshold)
              │
              └─ liveness=true
                        │
                        ▼
               ensure_face_quality(resolved quality knobs)
                        │ fail → ImageQualityError (422)
                        ▼
               MiniFASNet ensure_live(resolved liveness_threshold)
                        │ fail → LivenessFailedError (422)
                        ▼
               store / search / verify
```

### Endpoint → overrides

| Endpoint | `options` JSON keys | Other |
|----------|---------------------|-------|
| `POST /v1/faces` (enroll) | quality + `liveness_threshold` | `liveness` Form |
| `PUT /v1/faces/{face_id}` (update) | quality + `liveness_threshold` | `liveness` Form |
| `POST /v1/faces/search` | quality + `liveness_threshold` | `liveness` Form; Query `min_score`, `limit` |
| `POST /v1/faces/{face_id}/verify` | quality + `liveness_threshold` + `match_threshold` | `liveness` Form |

## 5. `options` JSON catalog

| JSON key | Settings default | Applied when |
|----------|------------------|--------------|
| `blur_min_variance` | `face_blur_min_variance` | `liveness=true` |
| `brightness_min` | `face_brightness_min` | `liveness=true` |
| `brightness_max` | `face_brightness_max` | `liveness=true` |
| `min_side_px` | `face_min_side_px` | `liveness=true` |
| `liveness_threshold` | `liveness_threshold` | `liveness=true` |
| `match_threshold` | `face_match_threshold` | verify only (ignored on other endpoints) |

**Resolve rule:** `effective = options.key if set else settings.<field>`.

**Errors:** invalid / unknown JSON → HTTP `422`. Quality and liveness failures unchanged (`ImageQualityError` / `LivenessFailedError`, both HTTP `422`).

## 6. Implementation shape

- `FaceRequestOptions` in `app/api/schemas.py`; `_parse_options` in `app/api/routes.py`.
- Endpoints take `options: str | None = Form(None)`; resolve into `_maybe_face_quality` / `_maybe_liveness` / verify match.
- Document in README / `.env.example` (env = server defaults; `options` = per-request).
- No new persistence, auth, or response schema fields.

## 7. Non-goals

- Server-enforced min/max bands or “stricter-only” overrides
- Flat Form floats for each knob (replaced by JSON `options`)
- Full `application/json` body with base64 image
- Moving search `min_score` into `options` (or deprecating Query)
- Client profiles, API-key-scoped defaults, or stored presets
- Quality checks when `liveness=false`
- Exposing grouping / ANN / upload-size / crop-scale knobs
- Changing MiniFASNet crop scale or model paths per request

## 8. Testing

- Omit `options` → behavior identical to current defaults
- `liveness=true` + stricter `blur_min_variance` in `options` → blurry image `422` quality
- `liveness=true` + custom `liveness_threshold` affects MiniFASNet pass/fail
- Verify with custom `match_threshold` → `match` and returned `threshold` use client value
- Invalid `options` JSON → `422`
- Search Query `min_score` unchanged
- Resolve helper: missing key → default; explicit `0` → disable blur check
