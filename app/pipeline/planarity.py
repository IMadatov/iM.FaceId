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
    """How foreshortened the face quad is, as a signed log edge ratio.

    Turning away from the camera shortens the far side of the eye/mouth
    quadrilateral and lengthens the near side. Because it reads only the four
    coplanar points — never the nose — it measures the *view*, not the subject,
    and so returns identical values for a real head and a photo at the same
    angle. That blindness is the point: it gates the planarity score without
    leaking any of the evidence that score depends on.
    """
    k = _as_kps(kps, label="kps")
    left = float(np.linalg.norm(k[LEFT_MOUTH] - k[LEFT_EYE]))
    right = float(np.linalg.norm(k[RIGHT_MOUTH] - k[RIGHT_EYE]))
    if left < 1e-6 or right < 1e-6:
        raise ValueError("degenerate landmarks: face quad has a zero-length edge")
    return float(np.log(left / right))


def view_change_spread(frames: Sequence[np.ndarray]) -> float:
    """Total range of viewing angle across frames — the motion gate for planarity.

    Without rotation a real face shows no parallax either, so a planarity score
    only carries information once this clears a minimum.
    """
    frames = list(frames)
    if not frames:
        raise ValueError("need at least one frame to measure view change")
    angles = [view_change(f) for f in frames]
    return float(max(angles) - min(angles))


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
