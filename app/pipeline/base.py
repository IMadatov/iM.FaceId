from typing import Protocol

import numpy as np


class FacePipeline(Protocol):
    def embed_bgr(self, image_bgr: np.ndarray) -> np.ndarray: ...
    def ready(self) -> bool: ...
