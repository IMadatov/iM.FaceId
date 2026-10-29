from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class FaceImageMetrics(Protocol):
    """The four OpenCV measurements the quality checks need.

    Behind a port so app/domain/ never imports cv2. The adapter calls the
    same cv2 functions the legacy ensure_face_quality did, so the numbers
    (and therefore the error strings) are unchanged.
    """

    def crop(self, image_bgr: np.ndarray, bbox: np.ndarray) -> np.ndarray: ...
    def gray(self, crop_bgr: np.ndarray) -> np.ndarray: ...
    def laplacian_variance(self, gray: np.ndarray) -> float: ...
    def mean_brightness(self, gray: np.ndarray) -> float: ...
