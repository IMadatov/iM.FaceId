# iM.FaceId — Client Face Parameter Overrides (Design)

**Date:** 2026-10-16  
**Status:** Approved  
**Project:** iM.FaceId  
**Parent:** [2026-10-05-im-faceid-webapi-design.md](./2026-10-05-im-faceid-webapi-design.md)  
**Related:** [2026-10-14-face-image-quality-gate-design.md](./2026-10-14-face-image-quality-gate-design.md)

## 1. Intent

Let API clients override face pipeline thresholds per request—image quality (blur, brightness, min face side), liveness score threshold, and verify match threshold—without changing server `.env` defaults.

**Success looks like:** omitting override fields preserves current behavior; sending Form floats changes effective thresholds for that call only; search continues to use existing Query `min_score` / `limit`.

## 2. Goals and success criteria

| Goal | Success |
|------|---------|
| Per-request overrides | Optional Form fields on enroll, update, search, verify |
| Defaults | Omit field → `Settings` / `.env` value |
| Full override | No server floors/ceilings; client may disable quality checks (same semantics as env: blur `0`, brightness `0`/`255`, min side `0`) |
| Scope of knobs | Quality + `liveness_threshold` on all liveness-capable image endpoints; `match_threshold` on verify; search score via existing Query `min_score` |
| Compatibility | Existing clients unchanged; search `min_score` stays Query |

## 3. Constraints (agreed)

| Decision | Choice |
|----------|--------|
| Transport | Optional multipart Form fields (same style as `liveness`) |
| Override policy | Full override (option A) |
| Endpoints | All image endpoints that already take `liveness` (option A) |
| Search score | Keep Query `min_score` / `limit`; do not duplicate as Form |
| Quality when `liveness=false` | Overrides accepted but ignored (quality still only runs on liveness path) |
| Response shape | No new fields; verify/search already return effective `threshold` |

## 4. Architecture

```
multipart Form (optional overrides) + Settings defaults
        │
        ▼
  resolve: client if not None else settings.*
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

| Endpoint | Form overrides | Other |
|----------|----------------|-------|
| `POST /v1/faces` (enroll) | quality + `liveness_threshold` | `liveness` |
| `PUT /v1/faces/{face_id}` (update) | quality + `liveness_threshold` | `liveness` |
| `POST /v1/faces/search` | quality + `liveness_threshold` | `liveness`; Query `min_score`, `limit` |
| `POST /v1/faces/{face_id}/verify` | quality + `liveness_threshold` + `match_threshold` | `liveness` |

## 5. Form field catalog

| Form field | Settings default | Applied when |
|------------|------------------|--------------|
| `blur_min_variance` | `face_blur_min_variance` | `liveness=true` |
| `brightness_min` | `face_brightness_min` | `liveness=true` |
| `brightness_max` | `face_brightness_max` | `liveness=true` |
| `min_side_px` | `face_min_side_px` | `liveness=true` |
| `liveness_threshold` | `liveness_threshold` | `liveness=true` |
| `match_threshold` | `face_match_threshold` | verify only |

**Resolve rule:** `effective = client if client is not None else settings.<field>`.

**Errors:** malformed types → FastAPI validation `422`. Quality and liveness failures unchanged (`ImageQualityError` / `LivenessFailedError`, both HTTP `422`).

## 6. Implementation shape

- Extend `app/api/routes.py` endpoints with optional `Form(None)` floats.
- Pass resolved thresholds into `_maybe_face_quality` and `_maybe_liveness` (and verify’s match check) as explicit arguments; `Settings` remains the default source.
- Document overrides in README / `.env.example` (env = server defaults; Form = per-request).
- No new persistence, auth, or response schema fields.

## 7. Non-goals

- Server-enforced min/max bands or “stricter-only” overrides
- Moving search `min_score` to Form (or deprecating Query)
- Client profiles, API-key-scoped defaults, or stored presets
- Quality checks when `liveness=false`
- Exposing grouping / ANN / upload-size / crop-scale knobs
- Changing MiniFASNet crop scale or model paths per request

## 8. Testing

- Omit all overrides → behavior identical to current defaults
- `liveness=true` + stricter `blur_min_variance` → blurry image `422` quality
- `liveness=true` + custom `liveness_threshold` affects MiniFASNet pass/fail
- Verify with custom `match_threshold` → `match` and returned `threshold` use client value
- Search Query `min_score` unchanged
- Resolve helper (if extracted): `None` → default; explicit `0` → disable blur check
