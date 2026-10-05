class InvalidImageError(Exception):
    """Bad or oversized image upload."""


class FaceDetectionError(Exception):
    """No face or multiple faces."""

    def __init__(self, message: str, *, code: str = "face_detection"):
        super().__init__(message)
        self.code = code


class FaceNotFoundError(Exception):
    """Unknown face_id."""


class LivenessFailedError(Exception):
    """Image failed anti-spoof / liveness check."""


class DependencyUnavailableError(Exception):
    """Model or Qdrant unavailable."""
