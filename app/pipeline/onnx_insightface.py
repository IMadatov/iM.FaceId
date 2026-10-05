import numpy as np

from app.errors import FaceDetectionError


class InsightFacePipeline:
    _dim: int = 512

    def __init__(
        self, model_dir: str, model_name: str = "buffalo_sc", embedding_dim: int = 512
    ) -> None:
        from insightface.app import FaceAnalysis

        self._dim = embedding_dim
        self._app = FaceAnalysis(
            name=model_name,
            root=model_dir,
            providers=["CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=-1, det_size=(640, 640))
        self._ready = True

    def ready(self) -> bool:
        return self._ready

    def embed_bgr(self, image_bgr: np.ndarray) -> np.ndarray:
        faces = self._app.get(image_bgr)
        if len(faces) == 0:
            raise FaceDetectionError("no face detected")
        if len(faces) > 1:
            raise FaceDetectionError("multiple faces detected")
        embedding = getattr(faces[0], "normed_embedding", None)
        if embedding is None:
            raise FaceDetectionError("face embedding unavailable")
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vec.shape != (self._dim,):
            raise FaceDetectionError(f"unexpected embedding shape {vec.shape}")
        if not np.all(np.isfinite(vec)):
            raise FaceDetectionError("embedding contains non-finite values")
        return vec
