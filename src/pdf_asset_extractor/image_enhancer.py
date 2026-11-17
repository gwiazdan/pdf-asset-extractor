"""Batch image enhancement primitives used before export."""

from dataclasses import dataclass
from enum import StrEnum
from io import BytesIO

from PIL import Image, ImageFilter


class EnhancementError(RuntimeError):
    """Raised when an image cannot be enhanced."""


class EnhancementMethod(StrEnum):
    """Supported image enhancement backends."""

    NONE = "none"
    LANCZOS = "lanczos"
    UNSHARP_MASK = "unsharp-mask"
    REAL_ESRGAN = "real-esrgan"


class ResizeMode(StrEnum):
    """Ways to determine the output dimensions."""

    FACTOR = "factor"
    WIDTH = "width"


@dataclass(frozen=True, slots=True)
class EnhancementOptions:
    """Configuration shared by all images in one export batch."""

    method: EnhancementMethod = EnhancementMethod.NONE
    resize_mode: ResizeMode = ResizeMode.FACTOR
    scale: float = 2.0
    target_width: int = 2000


def enhance_image(image_data: bytes, options: EnhancementOptions) -> bytes:
    """Apply the selected enhancement and return encoded image bytes."""
    if options.method is EnhancementMethod.NONE:
        return image_data

    if options.method is EnhancementMethod.REAL_ESRGAN:
        raise EnhancementError("Real-ESRGAN jest tymczasowo wyłączony.")

    try:
        with Image.open(BytesIO(image_data)) as source:
            image = source.copy()
            if options.method is EnhancementMethod.LANCZOS:
                size = _resize_dimensions(image, options)
                image = image.resize(size, Image.Resampling.LANCZOS)
            elif options.method is EnhancementMethod.UNSHARP_MASK:
                image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

            output = BytesIO()
            image.save(output, format="PNG")
            return output.getvalue()
    except (OSError, ValueError) as error:
        raise EnhancementError("Nie można poprawić wybranego obrazu.") from error


def _resize_dimensions(image: Image.Image, options: EnhancementOptions) -> tuple[int, int]:
    """Calculate dimensions without changing the source aspect ratio."""
    if options.resize_mode is ResizeMode.WIDTH:
        if options.target_width <= 0:
            raise ValueError("Docelowa szerokość musi być większa od zera.")
        width = options.target_width
        height = round(image.height * width / image.width)
        return width, max(height, 1)

    if options.scale <= 0:
        raise ValueError("Skala obrazu musi być większa od zera.")
    return (
        max(round(image.width * options.scale), 1),
        max(round(image.height * options.scale), 1),
    )


