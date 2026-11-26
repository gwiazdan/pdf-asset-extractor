"""Manual Lanczos resampling and unsharp mask implementations."""

import math

import numpy as np
from PIL import Image

SUPPORTED_MODES = frozenset({"L", "LA", "RGB", "RGBA"})
ALPHA_MODES = frozenset({"LA", "RGBA"})


def lanczos_resize(
    image: Image.Image,
    size: tuple[int, int],
    *,
    lobes: int = 3,
) -> Image.Image:
    """Resize an image using separable Lanczos resampling."""
    target_width, target_height = size
    if target_width <= 0 or target_height <= 0:
        raise ValueError("Docelowe wymiary obrazu muszą być większe od zera.")
    if lobes <= 0:
        raise ValueError("Parametr lobes filtra Lanczosa musi być większy od zera.")

    normalized = _normalize_image_mode(image)
    if normalized.size == (target_width, target_height):
        return normalized.copy()

    pixels = _to_float_channels(normalized)
    has_alpha = normalized.mode in ALPHA_MODES
    if has_alpha:
        alpha_scale = pixels[..., -1:] / 255.0
        pixels[..., :-1] *= alpha_scale

    src_height, src_width = pixels.shape[:2]
    # Pick the pass order with the smaller intermediate image.
    if target_height * src_width <= src_height * target_width:
        resized = (
            pixels
            if target_height == src_height
            else _resample_vertical(pixels, target_height, lobes)
        )
        resized = (
            resized
            if target_width == src_width
            else _resample_horizontal(resized, target_width, lobes)
        )
    else:
        resized = (
            pixels
            if target_width == src_width
            else _resample_horizontal(pixels, target_width, lobes)
        )
        resized = (
            resized
            if target_height == src_height
            else _resample_vertical(resized, target_height, lobes)
        )

    if has_alpha:
        alpha = np.clip(resized[..., -1:], 0.0, 255.0)
        alpha_scale = alpha / 255.0
        safe_scale = np.where(alpha_scale > 0.0, alpha_scale, 1.0)
        color = np.where(alpha_scale > 0.0, resized[..., :-1] / safe_scale, 0.0)
        resized = np.concatenate((color, alpha), axis=-1)

    return _from_float_channels(resized, normalized.mode)


def unsharp_mask(
    image: Image.Image,
    *,
    radius: float = 2.0,
    percent: int = 150,
    threshold: int = 3,
) -> Image.Image:
    """Sharpen an image with a manual Gaussian unsharp mask."""
    if radius < 0:
        raise ValueError("Promień maski wyostrzającej nie może być ujemny.")
    if percent < 0:
        raise ValueError("Siła maski wyostrzającej nie może być ujemna.")
    if threshold < 0:
        raise ValueError("Próg maski wyostrzającej nie może być ujemny.")

    normalized = _normalize_image_mode(image)
    if radius == 0 or percent == 0:
        return normalized.copy()

    pixels = _to_float_channels(normalized)
    has_alpha = normalized.mode in ALPHA_MODES
    color = pixels[..., :-1] if has_alpha else pixels

    blurred = _gaussian_blur(color, radius)
    detail = color - blurred
    strength = percent / 100.0
    if threshold > 0:
        sharpened_color = np.where(
            np.abs(detail) >= threshold,
            color + detail * strength,
            color,
        )
    else:
        sharpened_color = color + detail * strength

    if has_alpha:
        output = np.concatenate((sharpened_color, pixels[..., -1:]), axis=-1)
    else:
        output = sharpened_color

    return _from_float_channels(output, normalized.mode)


def _normalize_image_mode(image: Image.Image) -> Image.Image:
    """Convert palette and exotic image modes to channel layouts safe for math."""
    if image.mode in SUPPORTED_MODES:
        return image
    if image.mode == "1":
        return image.convert("L")
    if "A" in image.mode or "transparency" in image.info:
        return image.convert("RGBA")
    return image.convert("RGB")


def _to_float_channels(image: Image.Image) -> np.ndarray:
    """Return a contiguous float64 (H, W, C) view of the image pixels."""
    array = np.asarray(image, dtype=np.float64)
    if array.ndim == 2:
        return array[:, :, np.newaxis].copy()
    return array.copy()


def _from_float_channels(pixels: np.ndarray, mode: str) -> Image.Image:
    """Quantize a float64 (H, W, C) array back into a PIL image."""
    quantized = np.clip(np.rint(pixels), 0.0, 255.0).astype(np.uint8)
    if mode == "L":
        return Image.fromarray(quantized[:, :, 0], mode="L")
    return Image.fromarray(quantized, mode=mode)


def _sinc(values: np.ndarray) -> np.ndarray:
    """Evaluate normalized sinc(x) = sin(pi * x) / (pi * x)."""
    scaled = np.pi * values
    safe_scaled = np.where(values == 0.0, 1.0, scaled)
    return np.where(values == 0.0, 1.0, np.sin(scaled) / safe_scaled)


def _lanczos_kernel(distances: np.ndarray, lobes: int) -> np.ndarray:
    """Evaluate the windowed Lanczos-a kernel."""
    inside_support = np.abs(distances) < lobes
    kernel = _sinc(distances) * _sinc(distances / lobes)
    return np.where(inside_support, kernel, 0.0)


def _lanczos_weights(
    src_length: int,
    dst_length: int,
    lobes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute source indices and normalized Lanczos weights for one 1D axis."""
    ratio = src_length / dst_length
    filter_scale = max(ratio, 1.0)
    support = lobes * filter_scale

    centers = (np.arange(dst_length, dtype=np.float64) + 0.5) * ratio - 0.5
    left = np.floor(centers - support).astype(np.intp) + 1
    max_taps = math.ceil(2.0 * support) + 1
    offsets = np.arange(max_taps, dtype=np.intp)

    raw_indices = left[:, None] + offsets[None, :]
    valid = (raw_indices >= 0) & (raw_indices < src_length)
    clamped_indices = np.clip(raw_indices, 0, src_length - 1)

    distances = (raw_indices - centers[:, None]) / filter_scale
    weights = np.where(valid, _lanczos_kernel(distances, lobes), 0.0)

    weight_sums = weights.sum(axis=1, keepdims=True)
    zero_rows = weight_sums[:, 0] == 0.0
    if np.any(zero_rows):
        nearest = np.clip(np.rint(centers).astype(np.intp), 0, src_length - 1)
        weights[zero_rows] = 0.0
        clamped_indices[zero_rows] = nearest[zero_rows, None]
        weights[zero_rows, 0] = 1.0
        weight_sums[zero_rows] = 1.0

    weights /= weight_sums
    return clamped_indices, weights


def _resample_horizontal(
    pixels: np.ndarray,
    target_width: int,
    lobes: int,
) -> np.ndarray:
    """Apply 1D Lanczos resampling along image columns."""
    height, src_width, channels = pixels.shape
    indices, weights = _lanczos_weights(src_width, target_width, lobes)
    output = np.zeros((height, target_width, channels), dtype=np.float64)

    for tap in range(weights.shape[1]):
        tap_weights = weights[:, tap]
        if not np.any(tap_weights):
            continue
        output += pixels[:, indices[:, tap], :] * tap_weights[None, :, None]
    return output


def _resample_vertical(
    pixels: np.ndarray,
    target_height: int,
    lobes: int,
) -> np.ndarray:
    """Apply 1D Lanczos resampling along image rows."""
    src_height, width, channels = pixels.shape
    indices, weights = _lanczos_weights(src_height, target_height, lobes)
    output = np.zeros((target_height, width, channels), dtype=np.float64)

    for tap in range(weights.shape[1]):
        tap_weights = weights[:, tap]
        if not np.any(tap_weights):
            continue
        output += pixels[indices[:, tap], :, :] * tap_weights[:, None, None]
    return output


def _gaussian_kernel_1d(radius: float) -> np.ndarray:
    """Build a normalized 1D Gaussian kernel truncated at 3 sigma."""
    half_width = max(math.ceil(3.0 * radius), 1)
    offsets = np.arange(-half_width, half_width + 1, dtype=np.float64)
    weights = np.exp(-0.5 * np.square(offsets / radius))
    weights /= weights.sum()
    return weights


def _gaussian_blur(pixels: np.ndarray, radius: float) -> np.ndarray:
    """Blur an (H, W, C) array with a separable 1D Gaussian convolution."""
    height, width = pixels.shape[:2]
    kernel = _gaussian_kernel_1d(radius)
    half_width = len(kernel) // 2

    padded_x = np.pad(pixels, ((0, 0), (half_width, half_width), (0, 0)), mode="edge")
    blurred_x = np.zeros_like(pixels)
    for offset, weight in enumerate(kernel):
        blurred_x += padded_x[:, offset : offset + width, :] * weight

    padded_y = np.pad(blurred_x, ((half_width, half_width), (0, 0), (0, 0)), mode="edge")
    blurred = np.zeros_like(pixels)
    for offset, weight in enumerate(kernel):
        blurred += padded_y[offset : offset + height, :, :] * weight
    return blurred
