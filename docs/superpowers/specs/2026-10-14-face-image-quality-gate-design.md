# iM.FaceId — Face Image Quality Gate (Design)

**Date:** 2026-10-14  
**Status:** Approved  
**Project:** iM.FaceId  
**Parent:** [2026-10-05-im-faceid-webapi-design.md](./2026-10-05-im-faceid-webapi-design.md)

## 1. Intent

When clients request `liveness=true`, reject faces that are too blurry, poorly lit, or too small **before** MiniFASNet anti-spoof runs. Quality failures must be distinguishable from spoof failures so clients can prompt a retake instead of treating the capture as a presentation attack.

**Success looks like:** enroll/update/search/verify with `liveness=false` unchanged; with `liveness=true`, soft/dark/tiny faces return `422` with an explicit quality reason; thresholds tunable via `.env`.

## 2. Goals and success criteria

| Goal | Success |
|------|---------|
| Scope | Quality runs only when `liveness=true` |
| Checks | Blur (Laplacian variance), brightness (mean gray), face min side (px) |
| Tunable | All thresholds via Settings / env |
| Disable per check | Blur min `0`; brightness `0`/`255`; face min side `0` |
| Error separation | `ImageQualityError` ≠ `LivenessFailedError`; both HTTP `422` |
| Order | detect → quality → MiniFASNet → business logic |

## 3. Constraints (agreed)

| Decision | Choice |
|----------|--------|
| When | Only `liveness=true` path |
| Crop | Tight InsightFace bbox (not Silent-Face 2.7 expand) |
| Module | Separate `app/pipeline/image_quality.py` |
| Defaults | Blur min `50`; brightness `40`–`220`; min side `80` px |

## 4. Architecture

```
decode → detect_bgr
              │
              ├─ liveness=false → store / search / verify
              │
              └─ liveness=true
                        │
                        ▼
               ensure_face_quality(bbox crop)
                        │ fail → ImageQualityError (422)
                        ▼
               MiniFASNet ensure_live
                        │ fail → LivenessFailedError (422)
                        ▼
               store / search / verify
```

## 5. Configuration

| Env | Setting field | Default | Disable |
|-----|---------------|---------|---------|
| `FACE_BLUR_MIN_VARIANCE` | `face_blur_min_variance` | `50` | `0` |
| `FACE_BRIGHTNESS_MIN` | `face_brightness_min` | `40` | `0` |
| `FACE_BRIGHTNESS_MAX` | `face_brightness_max` | `220` | `255` |
| `FACE_MIN_SIDE_PX` | `face_min_side_px` | `80` | `0` |

## 6. Non-goals

- Dual-scale Silent-Face / second FAS model
- Video or challenge-response liveness
- Quality checks when `liveness=false`
- Changing MiniFASNet threshold defaults

## 7. Testing

- Unit tests for pass/fail/skip paths on synthetic crops
- API: `liveness=true` + blurry → `422` quality; same image without `liveness` still enrolls
