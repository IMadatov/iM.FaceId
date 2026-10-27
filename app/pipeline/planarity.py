"""Motion-parallax liveness: a flat replay is explained by a single homography.

Any two views of a *planar* surface are related exactly by a homography, however
the surface moves. A printed photo or a replayed screen is planar, so a homography
fitted to part of the face reprojects the rest of it with near-zero error. A real
face is not planar: the nose sits ~25mm off the plane through the eyes and mouth
corners, so under head rotation it lands somewhere the homography cannot predict.

We fit on the four near-coplanar InsightFace keypoints (both eyes, both mouth
corners) — exactly the four correspondences a homography needs — and measure how
far the predicted nose falls from the observed one. Normalising by interocular
distance makes the score independent of face size and camera distance.

The score only means something when the head actually turned: with no rotation a
real face produces no parallax either. Callers must gate on observed motion.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

LEFT_EYE = 0
RIGHT_EYE = 1
NOSE = 2
LEFT_MOUTH = 3
RIGHT_MOUTH = 4

_PLANE = [LEFT_EYE, RIGHT_EYE, LEFT_MOUTH, RIGHT_MOUTH]
_REQUIRED_POINTS = 5


def _as_kps(kps: np.ndarray, *, label: str) -> np.ndarray:
    arr = np.asarray(kps, dtype=np.float32).reshape(-1, 2)
    if arr.shape[0] != _REQUIRED_POINTS:
        raise ValueError(
            f"{label}: expected {_REQUIRED_POINTS} keypoints, got {arr.shape[0]}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label}: keypoints contain non-finite values")
    return arr


def planarity_residual(kps_a: np.ndarray, kps_b: np.ndarray) -> float:
    """Normalised nose reprojection error between two frames.

    Near zero means the two views are consistent with a flat surface (spoof);
    larger values mean the nose showed parallax, which only a 3D face produces.
    """
    a = _as_kps(kps_a, label="kps_a")
    b = _as_kps(kps_b, label="kps_b")

    interocular = float(np.linalg.norm(b[RIGHT_EYE] - b[LEFT_EYE]))
    if interocular < 1e-6:
        raise ValueError("degenerate landmarks: eyes coincide in the second frame")

    homography = cv2.getPerspectiveTransform(a[_PLANE], b[_PLANE])
    predicted = cv2.perspectiveTransform(
        a[NOSE].reshape(1, 1, 2), homography
    ).reshape(2)
    if not np.all(np.isfinite(predicted)):
        raise ValueError("degenerate landmarks: homography is not invertible")

    return float(np.linalg.norm(predicted - b[NOSE]) / interocular)


def view_change(kps: np.ndarray) -> float:
    """Face aspect ratio in log units — how foreshortened this view is.

    Yaw compresses the face horizontally: the interocular distance shrinks with
    the cosine of the angle while the eye-to-mouth span, lying along the axis of
    rotation, does not. Dividing the one by the other cancels both face size and
    pixel scale, so this tracks the *turn* itself.

    That matters because the obvious alternative — reading the near/far
    asymmetry of the quad's two vertical edges — is a pure perspective effect.
    It decays with camera distance, so it would gate the planarity score more
    tightly for a subject at arm's length than for one leaning into the lens,
    while the residual being gated barely moves with distance.

    The value carries a per-subject baseline (face proportions differ), so only
    differences between frames of one capture mean anything; see
    :func:`view_change_spread`. It is maximal at frontal and falls away for yaw
    in either direction, which makes it blind to turn *direction*.

    Reads only the four coplanar points, never the nose, so it returns identical
    values for a real head and a photo at the same angle. That blindness is the
    point: it gates the planarity score without leaking any of the evidence that
    score depends on.
    """
    k = _as_kps(kps, label="kps")
    interocular = float(np.linalg.norm(k[RIGHT_EYE] - k[LEFT_EYE]))
    eye_mid = (k[LEFT_EYE] + k[RIGHT_EYE]) / 2.0
    mouth_mid = (k[LEFT_MOUTH] + k[RIGHT_MOUTH]) / 2.0
    vertical = float(np.linalg.norm(mouth_mid - eye_mid))
    if interocular < 1e-6 or vertical < 1e-6:
        raise ValueError("degenerate landmarks: face quad has a zero-length edge")
    return float(np.log(interocular / vertical))


def _perspective_tilt(kps: np.ndarray) -> float:
    """Signed near/far asymmetry of the quad's two vertical edges.

    Turning shortens the far side of the eye/mouth quadrilateral and lengthens
    the near one. This is a pure perspective effect, so it fades as the subject
    steps back and is a poor measure of how far the head turned — but it is
    signed, and so carries the one thing :func:`view_change` cannot: which way.
    """
    k = _as_kps(kps, label="kps")
    left = float(np.linalg.norm(k[LEFT_MOUTH] - k[LEFT_EYE]))
    right = float(np.linalg.norm(k[RIGHT_MOUTH] - k[RIGHT_EYE]))
    if left < 1e-6 or right < 1e-6:
        raise ValueError("degenerate landmarks: face quad has a zero-length edge")
    return float(np.log(left / right))


def _spread(values: list[float]) -> float:
    return float(max(values) - min(values))


def view_change_spread(frames: Sequence[np.ndarray]) -> float:
    """Total range of view change across frames — the motion gate for planarity.

    Without rotation a real face shows no parallax either, so a planarity score
    only carries information once this clears a minimum.

    Takes the larger of two reads, because neither covers every capture alone.
    :func:`view_change` is the better measure of how far the head turned and
    holds up at any camera distance, but it is blind to direction: a capture of
    equal and opposite turns with no near-frontal frame — a left/right
    alternation, which is a natural way to film a liveness check — gives two
    mirror-image views of the same aspect ratio and so reads as no motion at
    all. :func:`_perspective_tilt` is signed and catches exactly that, at the
    cost of decaying with distance.

    Taking the max is safe because this gate only decides whether the planarity
    residual is *interpretable*, not whether the face is real. Over-reporting
    motion admits nothing: a flat or motionless capture still yields a near-zero
    residual and is turned away by the planarity threshold instead. It also
    means no capture that cleared the old perspective-only gate fails now.
    """
    frames = list(frames)
    if not frames:
        raise ValueError("need at least one frame to measure view change")
    return max(
        _spread([view_change(f) for f in frames]),
        _spread([_perspective_tilt(f) for f in frames]),
    )


def max_planarity_residual(frames: Sequence[np.ndarray]) -> float:
    """Largest pairwise residual across frames.

    A single pair showing parallax is enough to prove the subject is 3D, so the
    strongest pair decides — averaging would let a mostly-static sequence mask it.
    """
    frames = list(frames)
    if len(frames) < 2:
        raise ValueError("need at least two frames to measure parallax")
    return max(
        planarity_residual(frames[i], frames[j])
        for i in range(len(frames))
        for j in range(i + 1, len(frames))
    )
