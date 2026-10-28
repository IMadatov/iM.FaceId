import numpy as np

from app.errors import FaceDetectionError
from app.pipeline.base import DetectedFace


class InsightFacePipeline:
    _dim: int = 512

    def __init__(
        self,
        model_dir: str,
        model_name: str = "buffalo_sc",
        embedding_dim: int = 512,
        *,
        providers: list[str] | None = None,
        ctx_id: int = -1,
    ) -> None:
        from insightface.app import FaceAnalysis

        self._dim = embedding_dim
        self._app = FaceAnalysis(
            name=model_name,
            root=model_dir,
            providers=providers or ["CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        self._ready = True

    def ready(self) -> bool:
        return self._ready

    def detect_bgr(self, image_bgr: np.ndarray) -> DetectedFace:
        faces = self._app.get(image_bgr)
        if len(faces) == 0:
            raise FaceDetectionError("no face detected")
        if len(faces) > 1:
            raise FaceDetectionError("multiple faces detected")
        face = faces[0]
        embedding = getattr(face, "normed_embedding", None)
        if embedding is None:
            raise FaceDetectionError("face embedding unavailable")
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vec.shape != (self._dim,):
            raise FaceDetectionError(f"unexpected embedding shape {vec.shape}")
        if not np.all(np.isfinite(vec)):
            raise FaceDetectionError("embedding contains non-finite values")
        bbox = getattr(face, "bbox", None)
        if bbox is None:
            raise FaceDetectionError("face bbox unavailable")
        box = np.asarray(bbox, dtype=np.float32).reshape(-1)
        if box.shape != (4,):
            raise FaceDetectionError(f"unexpected bbox shape {box.shape}")
        if not np.all(np.isfinite(box)):
            raise FaceDetectionError("bbox contains non-finite values")
        return DetectedFace(embedding=vec, bbox=box, kps=self._keypoints(face))

    @staticmethod
    def _keypoints(face) -> np.ndarray | None:
        kps = getattr(face, "kps", None)
        if kps is None:
            return None
        points = np.asarray(kps, dtype=np.float32)
        if points.shape != (5, 2):
            raise FaceDetectionError(f"unexpected keypoint shape {points.shape}")
        if not np.all(np.isfinite(points)):
            raise FaceDetectionError("keypoints contain non-finite values")
        return points

    def embed_bgr(self, image_bgr: np.ndarray) -> np.ndarray:
        return self.detect_bgr(image_bgr).embedding
