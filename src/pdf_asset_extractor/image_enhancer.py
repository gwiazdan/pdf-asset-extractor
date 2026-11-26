"""Batch image enhancement primitives used before export."""

from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache, partial
from io import BytesIO
from typing import Any, Protocol, cast

import numpy as np
from PIL import Image

from pdf_asset_extractor.image_processing import lanczos_resize, unsharp_mask


class EnhancementError(RuntimeError):
    """Raised when an image cannot be enhanced."""


class EnhancementOutOfMemoryError(EnhancementError):
    """Raised when a super-resolution batch exceeds available GPU memory."""


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


class _SuperImageModel(Protocol):
    """Minimal callable interface satisfied by pretrained super-image models."""

    def __call__(self, inputs: Any) -> Any: ...


SUPER_IMAGE_MODELS: tuple[tuple[str, str], ...] = (
    ("drln-bam", "DrlnModel"),
    ("edsr", "EdsrModel"),
    ("msrn", "MsrnModel"),
)


@dataclass(frozen=True, slots=True)
class EnhancementOptions:
    """Configuration shared by all images in one export batch."""

    method: EnhancementMethod = EnhancementMethod.NONE
    resize_mode: ResizeMode = ResizeMode.FACTOR
    scale: float = 2.0
    target_width: int = 2000
    model_name: str = "edsr"
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
            if options.method is EnhancementMethod.LANCZOS:
                size = _resize_dimensions(source, options)
                image = lanczos_resize(source, size)
            elif options.method is EnhancementMethod.UNSHARP_MASK:
                image = unsharp_mask(
                    source,
                    radius=options.unsharp_radius,
                    percent=options.unsharp_percent,
                    threshold=options.unsharp_threshold,
                )
            else:
                image = source

            output = BytesIO()
            image.save(output, format="PNG")
            return output.getvalue()
    except (OSError, ValueError) as error:
        raise EnhancementError("Nie można poprawić wybranego obrazu.") from error


def enhance_images(
    image_data_batch: Sequence[bytes],
    options: EnhancementOptions,
    *,
    workers: int = 1,
) -> list[bytes]:
    """Enhance a batch of images, using threads for CPU-only methods."""
    if workers <= 0:
        raise ValueError("workers must be greater than zero")
    if not image_data_batch:
        return []
    if options.method is EnhancementMethod.NONE:
        return list(image_data_batch)
    if options.method is EnhancementMethod.SUPER_IMAGE:
        return _enhance_with_super_image_batch(image_data_batch, options)
    if workers == 1:
        return [enhance_image(image_data, options) for image_data in image_data_batch]

    # NumPy array ops in image_processing release the GIL, so threads scale on CPU batches.
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(
            executor.map(
                partial(enhance_image, options=options),
                image_data_batch,
            )
        )


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
    """Upscale one image through the batch inference implementation."""
    return _enhance_with_super_image_batch((image_data,), options)[0]


def _enhance_with_super_image_batch(
    image_data_batch: Sequence[bytes], options: EnhancementOptions
) -> list[bytes]:
    """Upscale images with one cached super-image model, batching same-sized inputs."""
    if options.scale not in {2.0, 3.0, 4.0}:
        raise EnhancementError("super-image obsługuje skale modeli 2x, 3x lub 4x.")

    # Lazy import avoids pulling PyTorch into memory when users only run CPU filters.
    try:
        from super_image import ImageLoader
        import torch
    except ImportError as error:
        raise EnhancementError(
            "Brak biblioteki super-image. Zainstaluj ją poleceniem: uv sync."
        ) from error

    if not image_data_batch:
        return []

    device = _resolve_device(options.device, torch.cuda.is_available())
    model = _load_super_image_model(options.model_name, int(options.scale), device)
    try:
        with ExitStack() as stack:
            source_images = [
                stack.enter_context(Image.open(BytesIO(image_data)))
                for image_data in image_data_batch
            ]
            rgb_images = [
                stack.enter_context(source_image.convert("RGB")) for source_image in source_images
            ]

            # Group by dimensions so heterogeneous PDF selections can still batch safely.
            size_groups: dict[tuple[int, int], list[int]] = defaultdict(list)
            for index, rgb_image in enumerate(rgb_images):
                size_groups[rgb_image.size].append(index)

            encoded_images: list[bytes | None] = [None] * len(image_data_batch)
            for indices in size_groups.values():
                group_tensors = [
                    # Upstream super_image mis-annotates load_image(image: PIL.Image) as module.
                    ImageLoader.load_image(cast(Any, rgb_images[index]))
                    for index in indices
                ]
                inputs = torch.cat(group_tensors, dim=0).to(device)
                with torch.inference_mode():
                    output = model(inputs)
                if not isinstance(output, torch.Tensor) or output.shape[0] != len(indices):
                    raise ValueError("The selected model returned an invalid batch.")

                output_array = output.clamp(0, 1).detach().cpu().numpy().transpose(0, 2, 3, 1)
                for original_index, image_array in zip(indices, output_array, strict=True):
                    enhanced_pixels = np.rint(image_array * 255.0).astype(np.uint8)
                    enhanced_image = Image.fromarray(enhanced_pixels)
                    if options.resize_mode is ResizeMode.WIDTH:
                        target_size = _resize_dimensions(enhanced_image, options)
                        enhanced_image = lanczos_resize(enhanced_image, target_size)

                    encoded = BytesIO()
                    enhanced_image.save(encoded, format="PNG")
                    encoded_images[original_index] = encoded.getvalue()

            return [payload for payload in encoded_images if payload is not None]
    except torch.cuda.OutOfMemoryError as error:
        raise EnhancementOutOfMemoryError(
            "The super-image batch exceeded available CUDA memory."
        ) from error
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise EnhancementError("Nie można poprawić obrazu przez super-image.") from error


def _resolve_device(requested_device: str, cuda_available: bool) -> str:
    """Resolve the requested accelerator while keeping CPU as a safe fallback."""
    if requested_device == "cpu":
        return "cpu"
    if requested_device == "cuda" and not cuda_available:
        raise EnhancementError("Wybrano CUDA, ale PyTorch nie wykrywa GPU CUDA.")
    if requested_device == "auto":
        return "cuda" if cuda_available else "cpu"
    return requested_device


@lru_cache(maxsize=1)
def _load_super_image_model(model_name: str, scale: int, device: str) -> _SuperImageModel:
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
