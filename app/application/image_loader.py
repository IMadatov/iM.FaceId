import base64

import numpy as np

from app.application.params import ImageLimits
from app.domain.errors import InvalidImageError
from app.domain.ports.imaging import ImageDecoder


class Base64ImageLoader:
    """Base64 (optionally data-URL wrapped) -> BGR ndarray.

    Lifted verbatim from the old routes._decode_base64_image, with the
    Settings lookup replaced by an ImageLimits argument.
    """

    def __init__(self, decoder: ImageDecoder) -> None:
        self._decoder = decoder

    def load(self, image_base64: str, *, limits: ImageLimits) -> np.ndarray:
        raw = image_base64.strip()
        if raw.startswith("data:") and "," in raw:
            raw = raw.split(",", 1)[1]
        try:
            data = base64.b64decode(raw, validate=False)
        except Exception as exc:
            raise InvalidImageError("invalid image_base64") from exc
        return self._decoder.decode(
            data, max_side=limits.max_side, max_bytes=limits.max_bytes
        )
