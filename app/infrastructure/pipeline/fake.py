import numpy as np

from app.domain.errors import FaceDetectionError
from app.domain.face import DetectedFace


# A real head alternating between a hard left and a hard right turn: one adult
# face model (25mm nose) pinhole-projected at -25 and +25 degrees of yaw.
#
# Multi-frame liveness gates on observed motion, so the stand-in pipeline has to
# move -- consecutive detections must differ in viewing angle, whatever the
# frame count, and alternating two poses satisfies that for any sequence length.
# The protruding nose gives the pair the parallax a flat replay cannot produce.
# tests/synthetic_face.py regenerates these: turning_head([-25.0, 25.0]).
_DEFAULT_KPS_FRAMES = (
    np.array(
        [
            [286.52, 215.38],
            [351.82, 216.59],
            [333.28, 252.57],
            [292.22, 289.04],
            [346.63, 287.01],
        ],
        dtype=np.float32,
    ),
    np.array(
        [
            [288.18, 216.59],
            [353.48, 215.38],
            [306.72, 252.57],
            [293.37, 287.01],
            [347.78, 289.04],
        ],
        dtype=np.float32,
    ),
)


class FakeFacePipeline:
    def __init__(
        self, faces_count: int = 1, kps_frames: list | None = None
    ) -> None:
        self.faces_count = faces_count
        # Cycled per detect call, so frame i of a request sees keypoints i.
        self._kps_frames = (
            list(_DEFAULT_KPS_FRAMES) if kps_frames is None else list(kps_frames)
        )
        self._detect_calls = 0

    def ready(self) -> bool:
        return True

    def detect_bgr(self, image_bgr: np.ndarray) -> DetectedFace:
        if self.faces_count == 0 or float(image_bgr.mean()) < 1.0:
            raise FaceDetectionError("no face detected")
        if self.faces_count > 1:
            raise FaceDetectionError("multiple faces detected")
        flat = image_bgr.reshape(-1).astype(np.float32)
        rng = np.random.default_rng(int(flat.sum()) % (2**32 - 1))
        v = rng.standard_normal(512, dtype=np.float32)
        v /= np.linalg.norm(v) + 1e-12
        h, w = image_bgr.shape[:2]
        # Center box covering half the frame (enough for Silent-Face-style crop)
        x1 = w * 0.25
        y1 = h * 0.25
        x2 = w * 0.75
        y2 = h * 0.75
        bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
        kps = self._kps_frames[self._detect_calls % len(self._kps_frames)]
        self._detect_calls += 1
        return DetectedFace(embedding=v, bbox=bbox, kps=kps)

    def embed_bgr(self, image_bgr: np.ndarray) -> np.ndarray:
        return self.detect_bgr(image_bgr).embedding
