"""Batch image enhancement primitives used before export."""

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
from functools import partial
from io import BytesIO

import numpy as np
from PIL import Image

from pdf_asset_extractor.image_processing import lanczos_resize, unsharp_mask
from pdf_asset_extractor.onnx_upscaler import (
    DEFAULT_TILE_OVERLAP,
    DEFAULT_TILE_SIZE,
    OnnxCancelledError,
    OnnxUpscalerError,
    OnnxUpscalerSession,
    load_onnx_upscaler,
    resolve_onnx_providers,
    upscale_rgb_array,
)

ProgressCallback = Callable[[str, int, int], None]
CancelCheck = Callable[[], bool]


class EnhancementError(RuntimeError):
    """Raised when an image cannot be enhanced."""


class EnhancementCancelledError(EnhancementError):
    """Raised when an active image enhancement operation is cancelled."""


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
    model_name: str = "msrn"
    device: str = "auto"
    unsharp_radius: float = 2.0
    unsharp_percent: int = 150
    unsharp_threshold: int = 3
    tile_size: int = DEFAULT_TILE_SIZE
    tile_overlap: int = DEFAULT_TILE_OVERLAP


def enhance_image(
    image_data: bytes,
    options: EnhancementOptions,
    *,
    progress_callback: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
) -> bytes:
    """Apply the selected enhancement and return encoded image bytes."""
    if cancel_check is not None and cancel_check():
        raise EnhancementCancelledError("Eksport został anulowany.")

    if options.method is EnhancementMethod.NONE:
        return image_data

    if options.method is EnhancementMethod.SUPER_IMAGE:
        return _enhance_with_super_image(
            image_data,
            options,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
        )

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
    progress_callback: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
) -> list[bytes]:
    """Enhance a batch of images, using threads for CPU-only methods."""
    if workers <= 0:
        raise ValueError("workers must be greater than zero")
    if not image_data_batch:
        return []
    if options.method is EnhancementMethod.NONE:
        return list(image_data_batch)
    if options.method is EnhancementMethod.SUPER_IMAGE:
        return _enhance_with_super_image_batch(
            image_data_batch,
            options,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
        )
    if workers == 1:
        return [
            enhance_image(
                image_data,
                options,
                progress_callback=progress_callback,
                cancel_check=cancel_check,
            )
            for image_data in image_data_batch
        ]

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


def _enhance_with_super_image(
    image_data: bytes,
    options: EnhancementOptions,
    *,
    progress_callback: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
) -> bytes:
    """Upscale one image through the ONNX super-resolution pipeline."""
    return _enhance_with_super_image_batch(
        (image_data,),
        options,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
    )[0]


def _load_super_image_model(
    model_name: str,
    scale: int,
    device: str,
    *,
    progress_callback: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
) -> OnnxUpscalerSession:
    """Load the cached ONNX super-resolution session, reporting download progress."""
    if model_name not in dict(SUPER_IMAGE_MODELS):
        raise EnhancementError(f"Nieznany model super-image: {model_name}")

    def _on_download_progress(downloaded: int, total: int | None) -> None:
        if progress_callback is None:
            return
        downloaded_mb = downloaded / (1024 * 1024)
        if total is not None and total > 0:
            total_mb = total / (1024 * 1024)
            percent = min(int(downloaded * 100 / total), 100)
            progress_callback(
                f"Pobieranie modelu {model_name} ({scale}x): "
                f"{downloaded_mb:.1f} / {total_mb:.1f} MB",
                percent,
                100,
            )
        else:
            progress_callback(
                f"Pobieranie modelu {model_name} ({scale}x): {downloaded_mb:.1f} MB...",
                0,
                0,
            )

    try:
        return load_onnx_upscaler(
            model_name,
            scale,
            device,
            download_progress_callback=_on_download_progress,
            cancel_check=cancel_check,
        )
    except OnnxCancelledError as error:
        raise EnhancementCancelledError(str(error)) from error
    except OnnxUpscalerError as error:
        raise EnhancementError(str(error)) from error


def _enhance_with_super_image_batch(
    image_data_batch: Sequence[bytes],
    options: EnhancementOptions,
    *,
    progress_callback: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
) -> list[bytes]:
    """Upscale images using cached ONNX Runtime super-resolution with overlap tiling."""
    if options.scale not in {2.0, 4.0}:
        raise EnhancementError("super-image obsługuje skale modeli 2x lub 4x.")
    if not image_data_batch:
        return []

    scale_int = int(options.scale)
    session = _load_super_image_model(
        options.model_name,
        scale_int,
        options.device,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
    )

    encoded_images: list[bytes] = []
    total_images = len(image_data_batch)
    try:
        for image_index, image_data in enumerate(image_data_batch, start=1):
            if cancel_check is not None and cancel_check():
                raise EnhancementCancelledError("Eksport został anulowany.")

            with (
                Image.open(BytesIO(image_data)) as source_image,
                source_image.convert("RGB") as rgb_image,
            ):
                rgb_pixels = np.asarray(rgb_image, dtype=np.uint8)

            def _on_tile_progress(
                tile_idx: int,
                total_tiles: int,
                *,
                current_image: int = image_index,
            ) -> None:
                if progress_callback is None:
                    return
                if total_tiles > 1:
                    label = (
                        f"Przetwarzanie ({options.model_name} {scale_int}x, "
                        f"{session.accelerator_label}) — kafelek {tile_idx}/{total_tiles}"
                    )
                else:
                    label = (
                        f"Przetwarzanie ({options.model_name} {scale_int}x, "
                        f"{session.accelerator_label})"
                    )
                if total_images > 1:
                    label = f"Obraz {current_image}/{total_images}: {label}"
                progress_callback(label, tile_idx, total_tiles)

            upscaled_pixels = upscale_rgb_array(
                rgb_pixels,
                runner=session,
                scale=scale_int,
                tile_size=options.tile_size,
                tile_overlap=options.tile_overlap,
                progress_callback=_on_tile_progress,
                cancel_check=cancel_check,
            )
            enhanced_image = Image.fromarray(upscaled_pixels, mode="RGB")
            if options.resize_mode is ResizeMode.WIDTH:
                target_size = _resize_dimensions(enhanced_image, options)
                enhanced_image = lanczos_resize(enhanced_image, target_size)

            encoded = BytesIO()
            enhanced_image.save(encoded, format="PNG")
            encoded_images.append(encoded.getvalue())

        return encoded_images
    except OnnxCancelledError as error:
        raise EnhancementCancelledError(str(error)) from error
    except EnhancementCancelledError:
        raise
    except MemoryError as error:
        raise EnhancementOutOfMemoryError(
            "Brak pamięci podczas poprawy obrazu przez super-image."
        ) from error
    except (OnnxUpscalerError, OSError, RuntimeError, TypeError, ValueError) as error:
        message = str(error)
        if "out of memory" in message.lower() or "E_OUTOFMEMORY" in message:
            raise EnhancementOutOfMemoryError(
                f"Brak pamięci GPU/RAM podczas poprawy obrazu: {message}"
            ) from error
        raise EnhancementError(
            f"Nie można poprawić obrazu przez super-image: {message}"
        ) from error


def _resolve_device(
    requested_device: str,
    *,
    available_providers: Sequence[str] | None = None,
) -> tuple[list[str], str]:
    """Resolve the requested accelerator into ONNX Runtime execution providers."""
    try:
        return resolve_onnx_providers(requested_device, available_providers)
    except OnnxUpscalerError as error:
        raise EnhancementError(str(error)) from error
