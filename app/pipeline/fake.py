import numpy as np

from app.errors import FaceDetectionError
from app.pipeline.base import DetectedFace


class FakeFacePipeline:
    def __init__(self, faces_count: int = 1) -> None:
        self.faces_count = faces_count

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
        return DetectedFace(embedding=v, bbox=bbox)

    def embed_bgr(self, image_bgr: np.ndarray) -> np.ndarray:
        return self.detect_bgr(image_bgr).embedding
