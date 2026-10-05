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


class OnnxLivenessChecker:
    """MiniFASNet-V2: input (1,3,80,80) BGR/255 → softmax [live, print, replay].

    Expect a face-centered frame (same crop used for recognition). We resize the
    full frame to 80×80 — no Haar dependency (opencv-headless safe).
    """

    def __init__(self, model_path: str, *, crop_scale: float = 2.7) -> None:
        import onnxruntime as ort

        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"liveness model not found: {model_path}")
        self._session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        self._crop_scale = crop_scale  # reserved for future detector-based crop
        self._ready = True

    def ready(self) -> bool:
        return self._ready

    def _preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        resized = cv2.resize(image_bgr, (80, 80), interpolation=cv2.INTER_AREA)
        tensor = resized.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        return tensor

    def score_bgr(self, image_bgr: np.ndarray) -> float:
        inp = self._preprocess(image_bgr)
        outputs = self._session.run(None, {self._input_name: inp})
        probs = _softmax(np.asarray(outputs[0]))
        # Class 0 = live (Silent-Face / MiniFASNet-V2 convention)
        return float(probs[0])

    def ensure_live(self, image_bgr: np.ndarray, *, threshold: float) -> float:
        score = self.score_bgr(image_bgr)
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
