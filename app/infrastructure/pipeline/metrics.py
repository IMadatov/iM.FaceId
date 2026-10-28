"""Shared face-crop pixel metrics for the liveness/quality path."""

from __future__ import annotations

import cv2
import numpy as np


def _crop_face_bgr(image_bgr: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Tight crop from InsightFace-style ``[x1, y1, x2, y2]`` bbox."""
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in np.asarray(bbox, dtype=np.float64).reshape(-1)[:4])
    ix1 = max(0, min(w - 1, int(np.floor(x1))))
    iy1 = max(0, min(h - 1, int(np.floor(y1))))
    ix2 = max(ix1 + 1, min(w, int(np.ceil(x2))))
    iy2 = max(iy1 + 1, min(h, int(np.ceil(y2))))
    return image_bgr[iy1:iy2, ix1:ix2]


class OpenCvFaceImageMetrics:
    """FaceImageMetrics adapter. Thin wrappers, no logic of their own."""

    def crop(self, image_bgr: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        return _crop_face_bgr(image_bgr, bbox)

    def gray(self, crop_bgr: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    def laplacian_variance(self, gray: np.ndarray) -> float:
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def mean_brightness(self, gray: np.ndarray) -> float:
        return float(gray.mean())
