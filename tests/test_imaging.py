import numpy as np
import cv2
import pytest
from app.errors import InvalidImageError
from app.imaging import decode_and_resize


def _png_bytes(w: int, h: int) -> bytes:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def test_rejects_empty_bytes():
    with pytest.raises(InvalidImageError):
        decode_and_resize(b"", max_side=640, max_bytes=5_000_000)


def test_rejects_oversized_upload():
    data = _png_bytes(10, 10)
    with pytest.raises(InvalidImageError):
        decode_and_resize(data, max_side=640, max_bytes=10)


@pytest.mark.parametrize("w,h", [(5000, 1), (1, 5000)])
def test_extreme_aspect_ratio_does_not_collapse_to_zero(w, h):
    out = decode_and_resize(_png_bytes(w, h), max_side=640, max_bytes=5_000_000)
    assert out.shape[0] >= 1 and out.shape[1] >= 1


def test_resize_failure_raises_invalid_image(monkeypatch):
    def boom(*args, **kwargs):
        raise cv2.error("resize failed")

    monkeypatch.setattr(cv2, "resize", boom)
    with pytest.raises(InvalidImageError):
        decode_and_resize(_png_bytes(1280, 720), max_side=640, max_bytes=5_000_000)


def test_resizes_long_side_to_max():
    data = _png_bytes(1280, 720)
    out = decode_and_resize(data, max_side=640, max_bytes=5_000_000)
    assert max(out.shape[0], out.shape[1]) == 640
