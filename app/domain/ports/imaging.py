from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ImageDecoder(Protocol):
    def decode(self, data: bytes, *, max_side: int, max_bytes: int) -> np.ndarray: ...
