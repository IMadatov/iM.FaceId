import cv2
import numpy as np
import pytest

from app.errors import ImageQualityError
from app.pipeline.image_quality import ensure_face_quality


def _textured_face(size: int = 120, mean: int = 120) -> np.ndarray:
    """Sharp checker-like face crop suitable for Laplacian blur check."""
    img = np.full((size, size, 3), mean, dtype=np.uint8)
    img[::4, :] = np.clip(mean + 80, 0, 255)
    img[:, ::4] = np.clip(mean + 80, 0, 255)
    return img


def _full_frame_with_face(
    face: np.ndarray, *, x1: int = 40, y1: int = 40
) -> tuple[np.ndarray, np.ndarray]:
    h, w = face.shape[:2]
    canvas = np.full((y1 + h + 40, x1 + w + 40, 3), 30, dtype=np.uint8)
    canvas[y1 : y1 + h, x1 : x1 + w] = face
    bbox = np.array([x1, y1, x1 + w, y1 + h], dtype=np.float32)
    return canvas, bbox


def test_sharp_face_passes_default_thresholds():
    frame, bbox = _full_frame_with_face(_textured_face(120))
    ensure_face_quality(
        frame,
        bbox,
        blur_min_variance=50.0,
        brightness_min=40.0,
        brightness_max=220.0,
        min_side_px=80.0,
    )


def test_blurry_face_fails():
    sharp = _textured_face(120)
    blurry = cv2.GaussianBlur(sharp, (31, 31), 0)
    frame, bbox = _full_frame_with_face(blurry)
    with pytest.raises(ImageQualityError, match="blur"):
        ensure_face_quality(
            frame,
            bbox,
            blur_min_variance=50.0,
            brightness_min=40.0,
            brightness_max=220.0,
            min_side_px=80.0,
        )


def test_dark_face_fails():
    # Uniform dark crop (no bright stripes that lift the mean).
    face = np.full((120, 120, 3), 15, dtype=np.uint8)
    frame, bbox = _full_frame_with_face(face)
    with pytest.raises(ImageQualityError, match="brightness"):
        ensure_face_quality(
            frame,
            bbox,
            blur_min_variance=0.0,
            brightness_min=40.0,
            brightness_max=220.0,
            min_side_px=80.0,
        )


def test_bright_face_fails():
    face = _textured_face(120, mean=245)
    frame, bbox = _full_frame_with_face(face)
    with pytest.raises(ImageQualityError, match="brightness"):
        ensure_face_quality(
            frame,
            bbox,
            blur_min_variance=0.0,
            brightness_min=40.0,
            brightness_max=220.0,
            min_side_px=80.0,
        )


def test_tiny_face_fails():
    face = _textured_face(40)
    frame, bbox = _full_frame_with_face(face)
    with pytest.raises(ImageQualityError, match="face too small|min side|size"):
        ensure_face_quality(
            frame,
            bbox,
            blur_min_variance=0.0,
            brightness_min=0.0,
            brightness_max=255.0,
            min_side_px=80.0,
        )


def test_blur_disabled_when_min_variance_zero():
    sharp = _textured_face(120)
    blurry = cv2.GaussianBlur(sharp, (31, 31), 0)
    frame, bbox = _full_frame_with_face(blurry)
    ensure_face_quality(
        frame,
        bbox,
        blur_min_variance=0.0,
        brightness_min=40.0,
        brightness_max=220.0,
        min_side_px=80.0,
    )


def test_brightness_disabled_with_full_range():
    face = _textured_face(120, mean=10)
    frame, bbox = _full_frame_with_face(face)
    ensure_face_quality(
        frame,
        bbox,
        blur_min_variance=0.0,
        brightness_min=0.0,
        brightness_max=255.0,
        min_side_px=80.0,
    )


def test_min_side_disabled_when_zero():
    face = _textured_face(40)
    frame, bbox = _full_frame_with_face(face)
    ensure_face_quality(
        frame,
        bbox,
        blur_min_variance=0.0,
        brightness_min=0.0,
        brightness_max=255.0,
        min_side_px=0.0,
    )
