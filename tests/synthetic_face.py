"""Synthetic 5-point faces for liveness tests.

Projects a known 3D landmark model through a pinhole camera, so tests can state
"a real face turned 25 degrees" or "a photo of a face was tilted" and get exact
keypoints. Deliberately independent of app.pipeline.planarity: expectations must
not be derived with the code under test.

Landmark order matches InsightFace: left eye, right eye, nose, mouth corners.
"""

import numpy as np

CAMERA_DISTANCE_MM = 500.0
FOCAL_PX = 600.0
PRINCIPAL_POINT = (320.0, 240.0)
#: Typical adult nose projection off the eye/mouth plane.
REAL_NOSE_MM = 25.0


def face_model(nose_protrusion: float = REAL_NOSE_MM) -> np.ndarray:
    """5 landmarks in model mm. ``nose_protrusion=0`` gives a flat (printed) face."""
    return np.array(
        [
            [-30.0, -20.0, 0.0],
            [30.0, -20.0, 0.0],
            [0.0, 10.0, -nose_protrusion],
            [-25.0, 40.0, 0.0],
            [25.0, 40.0, 0.0],
        ]
    )


def project(
    points3d: np.ndarray,
    yaw_deg: float,
    *,
    f: float = FOCAL_PX,
    dist: float = CAMERA_DISTANCE_MM,
) -> np.ndarray:
    """Yaw-rotate about the vertical axis and pinhole-project to (5, 2) pixels.

    ``f`` scales the face in pixels; ``dist`` changes how strong the perspective
    foreshortening is. They are not interchangeable: only ``dist`` alters the
    *shape* of the projected face, so distance-invariance claims need ``dist``.
    """
    t = np.deg2rad(yaw_deg)
    rot = np.array(
        [
            [np.cos(t), 0.0, np.sin(t)],
            [0.0, 1.0, 0.0],
            [-np.sin(t), 0.0, np.cos(t)],
        ]
    )
    cam = points3d @ rot.T
    cam[:, 2] += dist
    cx, cy = PRINCIPAL_POINT
    return np.stack(
        [cx + f * cam[:, 0] / cam[:, 2], cy + f * cam[:, 1] / cam[:, 2]], axis=1
    ).astype(np.float32)


def turning_head(
    yaws: list[float],
    *,
    nose_protrusion: float = REAL_NOSE_MM,
    dist: float = CAMERA_DISTANCE_MM,
):
    """Keypoints for one subject seen at each yaw angle."""
    model = face_model(nose_protrusion)
    return [project(model, yaw, dist=dist) for yaw in yaws]


def flat_replay(yaws: list[float], *, dist: float = CAMERA_DISTANCE_MM):
    """Keypoints for a printed photo or screen tilted to each angle."""
    return turning_head(yaws, nose_protrusion=0.0, dist=dist)
