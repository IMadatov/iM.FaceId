"""Deprecated import path. Kept so existing tests and adapters keep working.

New code imports from `app.domain.errors`.
"""

from app.domain.errors import (
    DependencyUnavailableError,
    FaceDetectionError,
    FaceNotFoundError,
    ImageQualityError,
    InvalidImageError,
    LivenessFailedError,
)

__all__ = [
    "DependencyUnavailableError",
    "FaceDetectionError",
    "FaceNotFoundError",
    "ImageQualityError",
    "InvalidImageError",
    "LivenessFailedError",
]
