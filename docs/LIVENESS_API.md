# Multi-frame Liveness API

`POST /v1/faces/liveness`

Passive, multi-frame liveness check. The client sends **several frames** of the
same person captured in quick succession; the service decides whether they came
from a **live person** rather than a **static photo or screen**. It performs no
enrollment, stores nothing, and returns no face embedding — it is a pure
preflight/gate.

---

## How it works

Each frame is decoded, a single face is detected, a quality gate is applied, and
the existing passive anti-spoof model scores it. The service then combines **three
signals**:

1. **Mean anti-spoof score** — the average of the per-frame model scores must
   clear `liveness_threshold`. Averaging across frames smooths out single-frame
   noise.
2. **Inter-frame variation** — the **maximum pairwise cosine distance** between
   the frames' face embeddings must clear `liveness_min_distance`. A real face
   moves and changes subtly between frames; a static photo or screen produces
   near-identical frames (distance ≈ 0), so it is rejected here.
3. **Motion parallax (planarity)** — see below. This is the signal that
   separates a *turning head* from a *tilted photo*, which the first two cannot.

The request passes only when **all** conditions hold:

```
mean(per-frame scores) >= liveness_threshold
AND
max_pairwise_cosine_distance(embeddings) >= liveness_min_distance
AND
view_change_spread(keypoints) >= liveness_min_view_change
AND
max_pairwise_planarity_residual(keypoints) >= liveness_min_planarity
```

### The planarity check

Any two views of a **flat** surface are related exactly by a homography, no
matter how the surface moves. A printed photo and a phone screen are flat; a
real face is not — the nose sits ~25 mm off the plane through the eyes and mouth
corners.

So the service fits a homography to the four near-coplanar InsightFace keypoints
(both eyes, both mouth corners), uses it to predict where the **nose** should
land in the other frame, and measures how far off that prediction is, normalised
by interocular distance. A replayed photo reprojects its nose almost perfectly
(residual ≈ 0); a real head that turned does not.

The score is only meaningful once the head has actually turned — a frontal pair
produces no parallax even for a real face. `liveness_min_view_change` is that
gate. It reads only the four coplanar points, never the nose, so it measures the
*viewing angle* and is deliberately blind to whether the subject is 3D; a spoof
therefore fails on **planarity**, not on "no movement".

### How the motion gate measures a turn

Primarily by face **aspect ratio**: yaw compresses the face horizontally, so the
interocular distance shrinks with the cosine of the angle while the eye-to-mouth
span, lying along the axis of rotation, does not. The log ratio of the two
cancels face size and pixel scale, and — unlike a near/far perspective cue — it
does not fade as the subject steps back. That matters because the residual being
gated is itself almost distance-independent, so a distance-sensitive gate would
reject captures whose parallax evidence is perfectly good.

Aspect ratio cannot tell *which way* the head turned: two opposite turns are
mirror images with the same aspect. So a left/right alternation that never
passes through frontal would read as zero motion. For that shape of capture the
spread falls back on the signed near/far asymmetry of the quad's vertical edges,
and takes whichever of the two reads is larger. The fallback decays with camera
distance, which is why the alternating row below drops off at 1.5 m while the
one-sided rows do not.

Taking the larger read cannot admit a spoof: the gate only decides whether the
residual is *interpretable*. A flat or motionless capture still scores a
near-zero residual and is turned away by `liveness_min_planarity` instead.

**Client implication:** the frames must contain a real head turn of roughly 13°
or more. A straight → right → left → straight capture works well; three frontal
selfies do not and will be rejected with *not enough head movement*. Capturing a
near-frontal frame alongside the turned ones gives the widest margin, since it
lets the distance-independent aspect read do the work.

Reference values from a synthetic 25 mm-nose face model (a flat photo scores a
residual of 0.000 in every row):

| Capture | Real face residual | Spread @0.5 m | Spread @1.5 m |
|---------|--------------------|---------------|---------------|
| frontal → 10° | 0.077 | 0.019 | 0.015 |
| frontal → 15° | 0.117 | 0.035 | 0.035 |
| frontal → 20° | 0.159 | 0.062 | 0.062 |
| frontal → 25° | 0.203 | 0.098 | 0.098 |
| −15° / 0 / +15° | 0.235 | 0.056 | 0.035 |
| −25° / +25°, no frontal frame | 0.411 | 0.091 | 0.030 |

> These come from an idealised pinhole projection with exact landmarks. Real
> SCRFD keypoints jitter by a few pixels, which raises the flat-photo residual
> above zero — calibrate on real captures before tightening the default. The
> jitter matters more than it looks: the residual is normalised by interocular
> distance, and the service downscales every frame to `max_image_side` (640 px)
> before detection, so a distant face is measured from few pixels.

---

## Request

**Method / path:** `POST /v1/faces/liveness`
**Content-Type:** `application/json`

### Body

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `images_base64` | `string[]` | **yes** | — | 2 … `liveness_max_frames` frames (default max **5**). Each item is a base64-encoded image (PNG/JPEG). A `data:image/...;base64,` prefix is accepted and stripped. |
| `blur_min_variance` | `number` | no | server (`50.0`) | Minimum Laplacian variance of the face crop; lower = too blurry. `0` disables the check. |
| `brightness_min` | `number` | no | server (`40.0`) | Minimum mean brightness of the face crop. |
| `brightness_max` | `number` | no | server (`220.0`) | Maximum mean brightness of the face crop. |
| `min_side_px` | `number` | no | server (`80.0`) | Minimum face bounding-box side in pixels. `0` disables the check. |
| `liveness_threshold` | `number` | no | server (`0.50`) | Minimum acceptable **mean** anti-spoof score (0–1). |
| `liveness_min_distance` | `number` | no | server (`0.05`) | Minimum acceptable **max pairwise** cosine distance between frame embeddings. |
| `liveness_min_planarity` | `number` | no | server (`0.06`) | Minimum acceptable **max pairwise** nose-reprojection residual. `0` disables the planarity check (and the movement gate with it). |
| `liveness_min_view_change` | `number` | no | server (`0.025`) | Minimum head turn required before the planarity score is trusted. ≈ 13° of yaw, at any camera distance. |

Notes:
- There is **no `liveness` flag** — this endpoint always runs the full gate
  (quality + score + inter-frame distance + planarity).
- Unknown fields are **rejected** (`extra="forbid"`).
- Omitted / `null` optional fields fall back to the server defaults shown above.

### Example

```json
{
  "images_base64": [
    "iVBORw0KGgoAAAANSUhEUgAA...frame1...",
    "iVBORw0KGgoAAAANSUhEUgAA...frame2...",
    "iVBORw0KGgoAAAANSUhEUgAA...frame3..."
  ],
  "liveness_threshold": 0.6
}
```

---

## Response `200 OK`

| Field | Type | Description |
|-------|------|-------------|
| `ok` | `boolean` | Always `true` on a 200 (failures are returned as error status codes). |
| `liveness_score` | `number` | Mean per-frame anti-spoof score (0–1). |
| `frame_count` | `integer` | Number of frames evaluated. |
| `inter_frame_distance` | `number` | Maximum pairwise cosine distance between frame embeddings. |
| `planarity_residual` | `number \| null` | Maximum pairwise nose-reprojection residual. `null` when the planarity check was disabled. Higher = more clearly 3D. |

```json
{
  "ok": true,
  "liveness_score": 0.97,
  "frame_count": 3,
  "inter_frame_distance": 0.11,
  "planarity_residual": 0.16
}
```

---

## Errors

All failures return a JSON body `{ "detail": "<message>" }`.

| Status | Condition | Example `detail` |
|--------|-----------|------------------|
| `400` | A frame is not valid/decodable base64, or exceeds the size limit | `frame[0]: invalid image_base64` |
| `422` | Fewer than 2 or more than `liveness_max_frames` frames; or an unknown field | `too many frames (max 5)` |
| `422` | No face, or multiple faces, in a frame | `frame[1]: no face detected` |
| `422` | A frame fails the quality gate (blur / brightness / size) | `image quality failed: face too small (...)` |
| `422` | Mean score below threshold | `liveness check failed (mean score=0.12 < 0.50)` |
| `422` | Frames too similar (static photo/screen suspected) | `liveness check failed (frames too similar: distance=0.00 < 0.05)` |
| `422` | Head did not turn enough to judge parallax | `liveness check failed (not enough head movement: view change=0.01 < 0.025)` |
| `422` | Nose showed no parallax (photo/screen suspected) | `liveness check failed (face looks flat: parallax=0.00 < 0.06)` |
| `503` | Liveness model not installed on the server | `liveness model not available (run scripts/download_liveness_model.py)` |
| `503` | Face model pack provides no keypoints | `planarity check needs facial keypoints, which this model pack does not provide (set LIVENESS_MIN_PLANARITY=0 to disable)` |

Error messages that originate in a specific frame are prefixed with
`frame[<index>]:` (0-based) so the client can tell which frame failed.

---

## Examples

### curl

```bash
# Build a JSON body from two image files and POST it.
python - <<'PY' > body.json
import base64, json
frames = ["frame1.png", "frame2.png", "frame3.png"]
imgs = [base64.b64encode(open(f, "rb").read()).decode() for f in frames]
print(json.dumps({"images_base64": imgs}))
PY

curl -sS -X POST http://localhost:8000/v1/faces/liveness \
  -H 'Content-Type: application/json' \
  --data @body.json
```

### Python (requests)

```python
import base64, requests

def b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")

resp = requests.post(
    "http://localhost:8000/v1/faces/liveness",
    json={
        "images_base64": [b64("frame1.png"), b64("frame2.png"), b64("frame3.png")],
        "liveness_threshold": 0.6,
    },
    timeout=30,
)

if resp.status_code == 200:
    body = resp.json()
    print("live:", body["liveness_score"], "distance:", body["inter_frame_distance"])
else:
    print("rejected:", resp.status_code, resp.json()["detail"])
```

---

## Server configuration

These `Settings` fields (env-overridable, see `.env.example`) control the
defaults:

| Setting | Default | Meaning |
|---------|---------|---------|
| `liveness_max_frames` | `5` | Upper bound on `images_base64` length. |
| `liveness_threshold` | `0.50` | Default mean-score threshold (shared with the single-image paths). |
| `liveness_min_distance` | `0.05` | Default inter-frame distance threshold. |
| `liveness_min_planarity` | `0.06` | Default nose-parallax threshold. `0` disables the planarity check. |
| `liveness_min_view_change` | `0.025` | Minimum observed head turn (≈ 13° of yaw, distance-independent) before planarity is judged. |

> **`liveness_min_distance` needs tuning.** `0.05` is a conservative starting
> value, not a calibrated one. Measure the max-pairwise distance on real device
> captures (genuine users vs. held-up photos/screens) and adjust: too low lets
> near-static spoofs through, too high falsely rejects very still users.

> **`liveness_min_planarity` needs tuning too.** `0.06` sits below the 0.077 a
> synthetic face produces at ±10° of yaw, but that model assumes pixel-perfect
> landmarks. Collect real genuine-vs-spoof captures, read `planarity_residual`
> off the 200 responses, and set the threshold between the two distributions.

---

## Scope & limitations

- **Minimum 2 frames.** The inter-frame signal cannot detect a static photo from
  a single frame, so the endpoint requires at least two.
- **Head turn required.** The planarity signal needs genuine rotation. Capture
  the frames as a guided turn (e.g. straight → right → left), not as three
  frontal shots.
- **Video replay and 3D masks remain out of scope.** A high-fidelity video
  replay varies between frames, so it can pass the distance gate — but because
  the replay is shown on a *flat* screen, the planarity check does reject it as
  long as the replayed subject turns. A **3D mask or a curved/bent print**
  defeats planarity, since it genuinely is not flat.
- **Frames are not checked to be the same person.** The gate only verifies that
  the frames are live and sufficiently varied; it does not bind them to one
  identity. Pair this with enroll/verify if identity matching is required.
- **No persistence.** Nothing is written to the vector store; no `face_id` or
  embedding is returned.
