"""Face-crop quality checks (blur, brightness, size) for the liveness path."""

from __future__ import annotations

import cv2
import numpy as np

from app.errors import ImageQualityError


def _crop_face_bgr(image_bgr: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Tight crop from InsightFace-style ``[x1, y1, x2, y2]`` bbox."""
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in np.asarray(bbox, dtype=np.float64).reshape(-1)[:4])
    ix1 = max(0, min(w - 1, int(np.floor(x1))))
    iy1 = max(0, min(h - 1, int(np.floor(y1))))
    ix2 = max(ix1 + 1, min(w, int(np.ceil(x2))))
    iy2 = max(iy1 + 1, min(h, int(np.ceil(y2))))
    return image_bgr[iy1:iy2, ix1:ix2]


def ensure_face_quality(
    image_bgr: np.ndarray,
    bbox: np.ndarray,
    *,
    blur_min_variance: float,
    brightness_min: float,
    brightness_max: float,
    min_side_px: float,
) -> None:
    """Raise ``ImageQualityError`` if the face crop fails quality checks.

    Disable a check by relaxing its threshold: ``blur_min_variance=0``,
    brightness ``0``/``255``, or ``min_side_px=0``.
    """
    x1, y1, x2, y2 = (float(v) for v in np.asarray(bbox, dtype=np.float64).reshape(-1)[:4])
    side = min(max(0.0, x2 - x1), max(0.0, y2 - y1))
    if min_side_px > 0 and side < min_side_px:
        raise ImageQualityError(
            f"image quality failed: face too small "
            f"(min side={side:.1f}px < {min_side_px:.1f}px)"
        )

    crop = _crop_face_bgr(image_bgr, bbox)
    if crop.size == 0:
        raise ImageQualityError("image quality failed: empty face crop")

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    if blur_min_variance > 0:
        variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if variance < blur_min_variance:
            raise ImageQualityError(
                f"image quality failed: blur too high "
                f"(laplacian_var={variance:.2f} < {blur_min_variance:.2f})"
            )

    mean_brightness = float(gray.mean())
    if brightness_min > 0 and mean_brightness < brightness_min:
        raise ImageQualityError(
            f"image quality failed: brightness too low "
            f"(mean={mean_brightness:.1f} < {brightness_min:.1f})"
        )
    if brightness_max < 255 and mean_brightness > brightness_max:
        raise ImageQualityError(
            f"image quality failed: brightness too high "
            f"(mean={mean_brightness:.1f} > {brightness_max:.1f})"
        )
