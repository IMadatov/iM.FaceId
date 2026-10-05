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


def test_resizes_long_side_to_max():
    data = _png_bytes(1280, 720)
    out = decode_and_resize(data, max_side=640, max_bytes=5_000_000)
    assert max(out.shape[0], out.shape[1]) == 640
