"""Manual Lanczos resampling and unsharp mask implementations."""

import math
from typing import Literal

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
        # Premultiply RGB/L by alpha so transparent pixels cannot bleed dark fringes.
        alpha_scale = pixels[..., -1:] / np.float32(255.0)
        pixels[..., :-1] *= alpha_scale

    src_height, src_width = pixels.shape[:2]
    # Pick the pass order that minimizes intermediate pixel count, skipping unchanged axes.
    if target_height * src_width <= src_height * target_width:
        resized = (
            pixels
            if target_height == src_height
            else _resample_axis(pixels, target_height, axis=0, lobes=lobes)
        )
        resized = (
            resized
            if target_width == src_width
            else _resample_axis(resized, target_width, axis=1, lobes=lobes)
        )
    else:
        resized = (
            pixels
            if target_width == src_width
            else _resample_axis(pixels, target_width, axis=1, lobes=lobes)
        )
        resized = (
            resized
            if target_height == src_height
            else _resample_axis(resized, target_height, axis=0, lobes=lobes)
        )

    if has_alpha:
        alpha = np.clip(resized[..., -1:], np.float32(0.0), np.float32(255.0))
        alpha_scale = alpha / np.float32(255.0)
        safe_scale = np.where(alpha_scale > 0.0, alpha_scale, np.float32(1.0))
        color = np.where(alpha_scale > 0.0, resized[..., :-1] / safe_scale, np.float32(0.0))
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
    # Sharpen color/luminance only; leave transparency untouched to avoid halo artifacts.
    color = pixels[..., :-1] if has_alpha else pixels

    blurred = _gaussian_blur(color, radius)
    detail = color - blurred
    strength = np.float32(percent / 100.0)
    if threshold > 0:
        sharpened_color = np.where(
            np.abs(detail) >= threshold,
            color + detail * strength,
            color,
        )
    else:
        detail *= strength
        detail += color
        sharpened_color = detail

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
    """Return a writable float32 (H, W, C) array of image pixels."""
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 2:
        return array[:, :, np.newaxis].copy()
    return array.copy()


def _from_float_channels(pixels: np.ndarray, mode: str) -> Image.Image:
    """Quantize a floating-point (H, W, C) array back into a PIL image."""
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
    """Compute source indices and normalized float64 Lanczos weights for one 1D axis."""
    ratio = src_length / dst_length
    # Stretch the kernel when downscaling to suppress high-frequency aliasing.
    filter_scale = max(ratio, 1.0)
    support = lobes * filter_scale

    # Map destination pixel centers (j + 0.5) into source pixel-center coordinates.
    centers = (np.arange(dst_length, dtype=np.float64) + 0.5) * ratio - 0.5
    left = np.floor(centers - support).astype(np.intp) + 1
    max_taps = math.ceil(2.0 * support) + 1
    offsets = np.arange(max_taps, dtype=np.intp)

    raw_indices = left[:, None] + offsets[None, :]
    valid = (raw_indices >= 0) & (raw_indices < src_length)
    clamped_indices = np.clip(raw_indices, 0, src_length - 1)

    distances = (raw_indices - centers[:, None]) / filter_scale
    weights = np.where(valid, _lanczos_kernel(distances, lobes), 0.0)

    # Renormalize in-bounds weights so flat regions remain invariant near image borders.
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


def _resample_axis(
    pixels: np.ndarray,
    target_length: int,
    *,
    axis: Literal[0, 1],
    lobes: int,
) -> np.ndarray:
    """Apply 1D Lanczos resampling along rows (axis=0) or columns (axis=1)."""
    src_length = pixels.shape[axis]
    indices, weights = _lanczos_weights(src_length, target_length, lobes)
    # Weights are computed in float64 for accuracy and cast to float32 for fast accumulation.
    weights_f32 = weights.astype(np.float32)

    out_shape = list(pixels.shape)
    out_shape[axis] = target_length
    output = np.zeros(out_shape, dtype=np.float32)
    scratch = np.empty_like(output)
    weight_shape = (target_length, 1, 1) if axis == 0 else (1, target_length, 1)

    for tap in range(weights_f32.shape[1]):
        tap_weights = weights_f32[:, tap]
        if not np.any(tap_weights):
            continue
        np.take(pixels, indices[:, tap], axis=axis, out=scratch)
        scratch *= tap_weights.reshape(weight_shape)
        output += scratch
    return output


def _gaussian_kernel_1d(radius: float) -> np.ndarray:
    """Build a normalized float64 1D Gaussian kernel truncated at 3 sigma."""
    half_width = max(math.ceil(3.0 * radius), 1)
    offsets = np.arange(-half_width, half_width + 1, dtype=np.float64)
    weights = np.exp(-0.5 * np.square(offsets / radius))
    weights /= weights.sum()
    return weights


def _gaussian_blur(pixels: np.ndarray, radius: float) -> np.ndarray:
    """Blur an (H, W, C) float32 array with a separable 1D Gaussian convolution."""
    height, width = pixels.shape[:2]
    kernel = _gaussian_kernel_1d(radius).astype(np.float32)
    half_width = len(kernel) // 2
    scratch = np.empty_like(pixels)

    padded_x = np.pad(pixels, ((0, 0), (half_width, half_width), (0, 0)), mode="edge")
    blurred_x = np.zeros_like(pixels)
    for offset, weight in enumerate(kernel):
        np.multiply(padded_x[:, offset : offset + width, :], weight, out=scratch)
        blurred_x += scratch

    padded_y = np.pad(blurred_x, ((half_width, half_width), (0, 0), (0, 0)), mode="edge")
    blurred = np.zeros_like(pixels)
    for offset, weight in enumerate(kernel):
        np.multiply(padded_y[offset : offset + height, :, :], weight, out=scratch)
        blurred += scratch
    return blurred
