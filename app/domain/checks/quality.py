from dataclasses import dataclass

from app.domain.checks.base import Frame
from app.domain.errors import ImageQualityError


@dataclass(frozen=True)
class MinSizeCheck:
    """Reject faces whose shorter bbox side is below the floor."""

    min_side_px: float
    name: str = "min_side"

    def verify(self, frame: Frame) -> None:
        side = frame.min_side
        if side < self.min_side_px:
            raise ImageQualityError(
                f"image quality failed: face too small "
                f"(min side={side:.1f}px < {self.min_side_px:.1f}px)"
            )


@dataclass(frozen=True)
class FaceCropCheck:
    """Reject a zero-area crop.

    Unconditional: the legacy gate cropped before any threshold branch, so
    this fires even with every other threshold relaxed to off. Must stay in
    the chain always and must run before any check that reads frame.gray —
    cv2.cvtColor on an empty array raises cv2.error, not a domain error.
    """

    name: str = "face_crop"

    def verify(self, frame: Frame) -> None:
        if frame.crop.size == 0:
            raise ImageQualityError("image quality failed: empty face crop")


@dataclass(frozen=True)
class BlurCheck:
    min_variance: float
    name: str = "blur"

    def verify(self, frame: Frame) -> None:
        variance = frame.metrics.laplacian_variance(frame.gray)
        if variance < self.min_variance:
            raise ImageQualityError(
                f"image quality failed: blur too high "
                f"(laplacian_var={variance:.2f} < {self.min_variance:.2f})"
            )


@dataclass(frozen=True)
class BrightnessMinCheck:
    minimum: float
    name: str = "brightness_min"

    def verify(self, frame: Frame) -> None:
        mean = frame.metrics.mean_brightness(frame.gray)
        if mean < self.minimum:
            raise ImageQualityError(
                f"image quality failed: brightness too low "
                f"(mean={mean:.1f} < {self.minimum:.1f})"
            )


@dataclass(frozen=True)
class BrightnessMaxCheck:
    maximum: float
    name: str = "brightness_max"

    def verify(self, frame: Frame) -> None:
        mean = frame.metrics.mean_brightness(frame.gray)
        if mean > self.maximum:
            raise ImageQualityError(
                f"image quality failed: brightness too high "
                f"(mean={mean:.1f} > {self.maximum:.1f})"
            )
