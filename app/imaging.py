import cv2
import numpy as np
from app.errors import InvalidImageError


def decode_and_resize(data: bytes, *, max_side: int, max_bytes: int) -> np.ndarray:
    if not data:
        raise InvalidImageError("empty image")
    if len(data) > max_bytes:
        raise InvalidImageError("image exceeds max upload size")
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise InvalidImageError("cannot decode image")
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side > max_side:
        scale = max_side / float(long_side)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        try:
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        except cv2.error as exc:
            raise InvalidImageError("cannot resize image") from exc
    return img
