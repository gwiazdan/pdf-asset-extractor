"""Batch image enhancement primitives used before export."""

from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from io import BytesIO

import numpy as np
from PIL import Image, ImageFilter


class EnhancementError(RuntimeError):
    """Raised when an image cannot be enhanced."""


class EnhancementMethod(StrEnum):
    """Supported image enhancement backends."""

    NONE = "none"
    LANCZOS = "lanczos"
    UNSHARP_MASK = "unsharp-mask"
    SUPER_IMAGE = "super-image"


class ResizeMode(StrEnum):
    """Ways to determine the output dimensions."""

    FACTOR = "factor"
    WIDTH = "width"


SUPER_IMAGE_MODELS: tuple[tuple[str, str], ...] = (
    ("edsr-base", "EdsrModel"),
    ("edsr", "EdsrModel"),
    ("msrn", "MsrnModel"),
    ("msrn-bam", "MsrnModel"),
    ("mdsr", "MdsrModel"),
    ("mdsr-bam", "MdsrModel"),
    ("drln-bam", "DrlnModel"),
    ("drln", "DrlnModel"),
    ("a2n", "A2nModel"),
    ("pan", "PanModel"),
    ("pan-bam", "PanModel"),
    ("carn", "CarnModel"),
    ("carn-bam", "CarnModel"),
    ("masa", "MasaModel"),
    ("jiif", "JiifModel"),
    ("liif", "LiifModel"),
    ("smsr", "SmsrModel"),
    ("rcan", "RcanModel"),
    ("physicssr", "PhysicssrModel"),
    ("han", "HanModel"),
    ("awsrn-bam", "AwsrnModel"),
    ("rnan", "RnanModel"),
    ("ddbpn", "DdbpnModel"),
)


@dataclass(frozen=True, slots=True)
class EnhancementOptions:
    """Configuration shared by all images in one export batch."""

    method: EnhancementMethod = EnhancementMethod.NONE
    resize_mode: ResizeMode = ResizeMode.FACTOR
    scale: float = 2.0
    target_width: int = 2000
    model_name: str = "edsr-base"
    device: str = "auto"
    unsharp_radius: float = 2.0
    unsharp_percent: int = 150
    unsharp_threshold: int = 3


def enhance_image(image_data: bytes, options: EnhancementOptions) -> bytes:
    """Apply the selected enhancement and return encoded image bytes."""
    if options.method is EnhancementMethod.NONE:
        return image_data

    if options.method is EnhancementMethod.SUPER_IMAGE:
        return _enhance_with_super_image(image_data, options)

    try:
        with Image.open(BytesIO(image_data)) as source:
            image = source.copy()
            if options.method is EnhancementMethod.LANCZOS:
                size = _resize_dimensions(image, options)
                image = image.resize(size, Image.Resampling.LANCZOS)
            elif options.method is EnhancementMethod.UNSHARP_MASK:
                image = image.filter(
                    ImageFilter.UnsharpMask(
                        radius=options.unsharp_radius,
                        percent=options.unsharp_percent,
                        threshold=options.unsharp_threshold,
                    )
                )

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


def _enhance_with_super_image(image_data: bytes, options: EnhancementOptions) -> bytes:
    """Upscale one image with a cached super-image model."""
    if options.scale not in {2.0, 3.0, 4.0}:
        raise EnhancementError("super-image obsługuje skale modeli 2x, 3x lub 4x.")

    try:
        from super_image import ImageLoader
        import torch

        device = _resolve_device(torch, options.device)
        model = _load_super_image_model(options.model_name, int(options.scale), device)
        with Image.open(BytesIO(image_data)) as source:
            source_image = source.convert("RGB")
            inputs = ImageLoader.load_image(source_image).to(device)
            with torch.inference_mode():
                output = model(inputs).clamp(0, 1)
            output_array = output[0].detach().cpu().numpy().transpose(1, 2, 0)
            enhanced_image = Image.fromarray(np.uint8(output_array * 255))

            if options.resize_mode is ResizeMode.WIDTH:
                target_size = _resize_dimensions(enhanced_image, options)
                enhanced_image = enhanced_image.resize(
                    target_size, Image.Resampling.LANCZOS
                )

            encoded = BytesIO()
            enhanced_image.save(encoded, format="PNG")
            return encoded.getvalue()
    except ImportError as error:
        raise EnhancementError(
            "Brak biblioteki super-image. Zainstaluj ją poleceniem: uv sync."
        ) from error
    except (OSError, RuntimeError, ValueError) as error:
        raise EnhancementError("Nie można poprawić obrazu przez super-image.") from error


def _resolve_device(torch: object, requested_device: str) -> str:
    """Resolve the requested accelerator while keeping CPU as a safe fallback."""
    if requested_device == "cpu":
        return "cpu"
    cuda_available = torch.cuda.is_available()  # type: ignore[attr-defined]
    if requested_device == "cuda" and not cuda_available:
        raise EnhancementError("Wybrano CUDA, ale PyTorch nie wykrywa GPU CUDA.")
    return "cuda" if requested_device == "auto" and cuda_available else requested_device


@lru_cache(maxsize=8)
def _load_super_image_model(model_name: str, scale: int, device: str) -> object:
    """Load and cache one pretrained super-image model."""
    try:
        import super_image
        import torch
    except ImportError as error:
        raise EnhancementError(
            "Brak biblioteki super-image. Zainstaluj ją poleceniem: uv sync."
        ) from error

    model_class_name = dict(SUPER_IMAGE_MODELS).get(model_name)
    if model_class_name is None:
        raise EnhancementError(f"Nieznany model super-image: {model_name}")
    model_class = getattr(super_image, model_class_name)
    model = model_class.from_pretrained(f"eugenesiow/{model_name}", scale=scale)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    return model.to(device).eval()


