# iM.FaceId — Client Face Parameter Overrides (Design)

**Date:** 2026-10-16  
**Status:** Approved  
**Project:** iM.FaceId  
**Parent:** [2026-10-05-im-faceid-webapi-design.md](./2026-10-05-im-faceid-webapi-design.md)  
**Related:** [2026-10-14-face-image-quality-gate-design.md](./2026-10-14-face-image-quality-gate-design.md)

## 1. Intent

Let API clients override face pipeline thresholds per request—and send the image—via a single **`application/json`** body (Swagger Request body like a typical login API), without changing server `.env` defaults.

**Success looks like:** omitting optional keys preserves current behavior; sending JSON fields changes effective thresholds for that call only; Swagger shows a required JSON request body (not multipart).

## 2. Goals and success criteria

| Goal | Success |
|------|---------|
| Transport | `application/json` body on enroll, update, search, verify |
| Image | Required `image_base64` (raw base64 or `data:` URL) |
| Liveness flag | JSON `liveness` (default `false`) — not a separate Form field |
| Defaults | Omit a key → `Settings` / `.env` value |
| Full override | No server floors/ceilings; disable semantics unchanged |
| Scope of knobs | Quality + `liveness_threshold`; verify `match_threshold`; search `limit` / `min_score` in the same JSON body |
| Compatibility | Multipart `/v1/faces` uploads removed (breaking change) |

## 3. Constraints (agreed)

| Decision | Choice |
|----------|--------|
| Transport | JSON body only (Login-style Swagger) |
| Override policy | Full override |
| Endpoints | enroll, update, search, verify |
| Search score / limit | In JSON body (rebuild still uses Query) |
| Quality when `liveness=false` | Threshold keys accepted but ignored |
| Response shape | Unchanged; verify/search return effective `threshold` |
| Unknown keys | HTTP `422` (`extra=forbid`) |

## 4. Architecture

```
application/json body
  image_base64 + liveness + optional thresholds
        │
        ▼
  decode base64 → decode_and_resize
        │
        ▼
detect_bgr → (if liveness) quality → (if liveness) MiniFASNet → business logic
```

### Request models

| Endpoint | Body model | Extra fields |
|----------|------------|--------------|
| `POST /v1/faces` | `FaceImageRequest` | — |
| `PUT /v1/faces/{face_id}` | `FaceImageRequest` | — |
| `POST /v1/faces/search` | `FaceSearchRequest` | `limit`, `min_score` |
| `POST /v1/faces/{face_id}/verify` | `FaceVerifyRequest` | `match_threshold` |

## 5. JSON field catalog

| JSON key | Settings default | Applied when |
|----------|------------------|--------------|
| `image_base64` | (required) | always |
| `liveness` | `false` | always |
| `blur_min_variance` | `face_blur_min_variance` | `liveness=true` |
| `brightness_min` | `face_brightness_min` | `liveness=true` |
| `brightness_max` | `face_brightness_max` | `liveness=true` |
| `min_side_px` | `face_min_side_px` | `liveness=true` |
| `liveness_threshold` | `liveness_threshold` | `liveness=true` |
| `match_threshold` | `face_match_threshold` | verify only |
| `limit` / `min_score` | search defaults | search only |

**Resolve rule:** `effective = body.key if set else settings.<field>`.

## 6. Implementation shape

- Schemas: `FaceImageRequest`, `FaceVerifyRequest`, `FaceSearchRequest` in `app/api/schemas.py`
- Routes: `_decode_base64_image`, `_run_quality_and_liveness`; no multipart File/Form on image endpoints
- Docs: README curl with `Content-Type: application/json`

## 7. Non-goals

- Keeping multipart file upload alongside JSON
- Server-enforced min/max bands
- Client profiles / stored presets
- Quality when `liveness=false`
- Exposing grouping / ANN / upload-size / crop-scale knobs

## 8. Testing

- Omit optional keys → env defaults
- `liveness=true` + stricter blur / liveness_threshold → expected `422`
- Verify `match_threshold` reflected in response
- Unknown JSON field → `422`
- Empty / corrupt base64 → `400`
