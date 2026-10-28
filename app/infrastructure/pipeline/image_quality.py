"""Face-crop quality checks (blur, brightness, size) for the liveness path."""

from __future__ import annotations

import numpy as np

from app.infrastructure.pipeline.metrics import _crop_face_bgr

__all__ = ["ensure_face_quality", "_crop_face_bgr"]


def ensure_face_quality(
    image_bgr: np.ndarray,
    bbox: np.ndarray,
    *,
    blur_min_variance: float,
    brightness_min: float,
    brightness_max: float,
    min_side_px: float,
) -> None:
    """Deprecated. Runs the domain quality chain; kept for the legacy routes."""
    from app.application.params import GateParams
    from app.domain.checks.base import Frame
    from app.domain.checks.chains import build_quality_chain
    from app.domain.face import DetectedFace
    from app.infrastructure.pipeline.metrics import OpenCvFaceImageMetrics

    gate = GateParams(
        blur_min_variance=blur_min_variance,
        brightness_min=brightness_min,
        brightness_max=brightness_max,
        min_side_px=min_side_px,
        liveness_threshold=0.0,
    )
    frame = Frame(
        image_bgr=image_bgr,
        detected=DetectedFace(embedding=None, bbox=bbox, kps=None),
        metrics=OpenCvFaceImageMetrics(),
    )
    for check in build_quality_chain(gate):
        check.verify(frame)
