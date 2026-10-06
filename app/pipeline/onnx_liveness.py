"""Silent-Face-Anti-Spoofing MiniFASNet-V2 ONNX checker (CPU)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.errors import LivenessFailedError


def _softmax(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float64).reshape(-1)
    x = x - np.max(x)
    e = np.exp(x)
    return (e / e.sum()).astype(np.float32)


def crop_face_bgr(
    image_bgr: np.ndarray, bbox: np.ndarray, *, scale: float
) -> np.ndarray:
    """Expand ``bbox`` by ``scale`` (Silent-Face) and crop; clamp to image bounds.

    ``bbox`` is ``[x1, y1, x2, y2]`` in pixel coordinates (InsightFace style).
    """
    if scale < 1.0:
        raise ValueError(f"scale must be >= 1.0, got {scale}")
    src_h, src_w = image_bgr.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in np.asarray(bbox, dtype=np.float64).reshape(-1)[:4])
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    scale = min((src_h - 1) / box_h, min((src_w - 1) / box_w, scale))
    new_w = box_w * scale
    new_h = box_h * scale
    center_x = x1 + box_w / 2.0
    center_y = y1 + box_h / 2.0
    left = center_x - new_w / 2.0
    top = center_y - new_h / 2.0
    right = center_x + new_w / 2.0
    bottom = center_y + new_h / 2.0
    if left < 0:
        right -= left
        left = 0.0
    if top < 0:
        bottom -= top
        top = 0.0
    if right > src_w - 1:
        left -= right - (src_w - 1)
        right = float(src_w - 1)
    if bottom > src_h - 1:
        top -= bottom - (src_h - 1)
        bottom = float(src_h - 1)
    left = max(0.0, left)
    top = max(0.0, top)
    ix1 = int(left)
    iy1 = int(top)
    ix2 = int(right) + 1
    iy2 = int(bottom) + 1
    ix2 = min(src_w, max(ix1 + 1, ix2))
    iy2 = min(src_h, max(iy1 + 1, iy2))
    return image_bgr[iy1:iy2, ix1:ix2].copy()


class OnnxLivenessChecker:
    """MiniFASNet-V2: input (1,3,80,80) BGR float0–255 → softmax [spoof, live, spoof].

    Silent-Face labels: class 1 = real/live; 0 and 2 = attack (print/replay).
    Prefer a face ``bbox`` so we crop with ``crop_scale`` (Silent-Face 2.7) before
    resize. Without bbox, falls back to full-frame resize (legacy / tests).
    """

    _LIVE_CLASS = 1

    def __init__(self, model_path: str, *, crop_scale: float = 2.7) -> None:
        import onnxruntime as ort

        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"liveness model not found: {model_path}")
        self._session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        self._crop_scale = crop_scale
        self._ready = True

    def ready(self) -> bool:
        return self._ready

    def _preprocess(
        self, image_bgr: np.ndarray, bbox: np.ndarray | None = None
    ) -> np.ndarray:
        if bbox is not None:
            image_bgr = crop_face_bgr(image_bgr, bbox, scale=self._crop_scale)
        resized = cv2.resize(image_bgr, (80, 80), interpolation=cv2.INTER_AREA)
        # Silent-Face ToTensor: HWC→CHW float without /255 for this ONNX export
        tensor = resized.astype(np.float32)
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        return tensor

    def score_bgr(
        self, image_bgr: np.ndarray, bbox: np.ndarray | None = None
    ) -> float:
        inp = self._preprocess(image_bgr, bbox=bbox)
        outputs = self._session.run(None, {self._input_name: inp})
        probs = _softmax(np.asarray(outputs[0]))
        return float(probs[self._LIVE_CLASS])

    def ensure_live(
        self,
        image_bgr: np.ndarray,
        *,
        threshold: float,
        bbox: np.ndarray | None = None,
    ) -> float:
        score = self.score_bgr(image_bgr, bbox=bbox)
        if score < threshold:
            raise LivenessFailedError(
                f"liveness check failed (score={score:.4f} < {threshold:.4f})"
            )
        return score


def load_liveness_checker(model_path: str, *, crop_scale: float = 2.7):
    """Return OnnxLivenessChecker or UnavailableLivenessChecker."""
    from app.pipeline.fake_liveness import UnavailableLivenessChecker

    try:
        return OnnxLivenessChecker(model_path, crop_scale=crop_scale)
    except Exception:
        return UnavailableLivenessChecker()
