import numpy as np

from app.errors import FaceDetectionError


class FakeFacePipeline:
    def __init__(self, faces_count: int = 1) -> None:
        self.faces_count = faces_count

    def ready(self) -> bool:
        return True

    def embed_bgr(self, image_bgr: np.ndarray) -> np.ndarray:
        if self.faces_count == 0 or float(image_bgr.mean()) < 1.0:
            raise FaceDetectionError("no face detected")
        if self.faces_count > 1:
            raise FaceDetectionError("multiple faces detected")
        flat = image_bgr.reshape(-1).astype(np.float32)
        rng = np.random.default_rng(int(flat.sum()) % (2**32 - 1))
        v = rng.standard_normal(512, dtype=np.float32)
        v /= np.linalg.norm(v) + 1e-12
        return v
