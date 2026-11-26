"""Unit tests for manual Lanczos resampling and unsharp mask."""

import numpy as np
from PIL import Image, ImageFilter
import pytest

from pdf_asset_extractor.image_processing import (
    _gaussian_kernel_1d,
    _lanczos_kernel,
    lanczos_resize,
    unsharp_mask,
)


def _sample_rgb_image(width: int = 64, height: int = 48) -> Image.Image:
    """Create a smooth-plus-edge synthetic RGB test image."""
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    red = ((x_coords / max(width - 1, 1)) * 200 + 20).astype(np.float64)
    green = ((y_coords / max(height - 1, 1)) * 180 + 30).astype(np.float64)
    blue = np.where((x_coords > width // 3) & (y_coords > height // 3), 210.0, 40.0)
    pixels = np.stack((red, green, blue), axis=-1).astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def test_lanczos_kernel_properties() -> None:
    points = np.array([-3.5, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 3.5])
    values = _lanczos_kernel(points, lobes=3)

    assert values[5] == pytest.approx(1.0)
    assert values[4] == pytest.approx(values[6])
    for zero_index in (0, 1, 2, 3, 7, 8, 9, 10):
        assert values[zero_index] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    ("mode", "color"),
    [
        ("L", 137),
        ("LA", (137, 200)),
        ("RGB", (42, 128, 211)),
        ("RGBA", (42, 128, 211, 180)),
    ],
)
@pytest.mark.parametrize("target_size", [(32, 24), (8, 6)])
def test_lanczos_preserves_constant_images(
    mode: str,
    color: int | tuple[int, ...],
    target_size: tuple[int, int],
) -> None:
    source = Image.new(mode, (16, 12), color=color)

    resized = lanczos_resize(source, target_size)

    assert resized.mode == mode
    assert resized.size == target_size
    expected = np.asarray(Image.new(mode, target_size, color=color))
    np.testing.assert_array_equal(np.asarray(resized), expected)


@pytest.mark.parametrize("target_size", [(128, 96), (96, 72), (32, 24), (64, 30), (40, 48)])
def test_lanczos_matches_reference_closely(target_size: tuple[int, int]) -> None:
    source = _sample_rgb_image(64, 48)

    manual = lanczos_resize(source, target_size)
    reference = source.resize(target_size, Image.Resampling.LANCZOS)

    assert manual.size == target_size
    mae = float(
        np.mean(
            np.abs(np.asarray(manual, dtype=np.float64) - np.asarray(reference, dtype=np.float64))
        )
    )
    assert mae < 0.5


def test_lanczos_identity_size_is_exact() -> None:
    source = _sample_rgb_image(24, 20)

    resized = lanczos_resize(source, source.size)

    np.testing.assert_array_equal(np.asarray(resized), np.asarray(source))


def test_lanczos_premultiplies_alpha_to_prevent_dark_fringes() -> None:
    pixels = np.zeros((16, 16, 4), dtype=np.uint8)
    pixels[:, 8:, 0] = 255
    pixels[:, 8:, 3] = 255
    source = Image.fromarray(pixels, mode="RGBA")

    resized = lanczos_resize(source, (32, 16))
    result = np.asarray(resized)

    semi_transparent = (result[:, :, 3] > 20) & (result[:, :, 3] < 235)
    assert np.any(semi_transparent)
    assert np.all(result[semi_transparent, 0] == 255)
    assert np.all(result[semi_transparent, 1] == 0)
    assert np.all(result[semi_transparent, 2] == 0)


def test_lanczos_handles_palette_and_bilevel_images() -> None:
    palette_image = _sample_rgb_image(16, 16).convert("P")
    bilevel_image = Image.new("1", (16, 16), color=1)

    resized_palette = lanczos_resize(palette_image, (32, 32))
    resized_bilevel = lanczos_resize(bilevel_image, (32, 32))

    assert resized_palette.mode == "RGB"
    assert resized_palette.size == (32, 32)
    assert resized_bilevel.mode == "L"
    assert np.all(np.asarray(resized_bilevel) == 255)


@pytest.mark.parametrize(
    ("size", "lobes", "expected_message"),
    [
        ((0, 10), 3, "wymiary"),
        ((10, -2), 3, "wymiary"),
        ((10, 10), 0, "lobes"),
    ],
)
def test_lanczos_rejects_invalid_arguments(
    size: tuple[int, int],
    lobes: int,
    expected_message: str,
) -> None:
    source = Image.new("RGB", (8, 8))

    with pytest.raises(ValueError, match=expected_message):
        lanczos_resize(source, size, lobes=lobes)


def test_gaussian_kernel_is_symmetric_and_normalized() -> None:
    kernel = _gaussian_kernel_1d(2.0)

    assert kernel.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(kernel, kernel[::-1])
    assert kernel[len(kernel) // 2] == kernel.max()


def test_unsharp_mask_sharpens_step_edge() -> None:
    pixels = np.full((32, 32, 3), 80, dtype=np.uint8)
    pixels[:, 16:, :] = 180
    source = Image.fromarray(pixels, mode="RGB")

    sharpened = np.asarray(unsharp_mask(source, radius=2.0, percent=150, threshold=3))

    assert int(sharpened[16, 15, 0]) < 80
    assert int(sharpened[16, 16, 0]) > 180
    assert int(sharpened[16, 0, 0]) == 80
    assert int(sharpened[16, 31, 0]) == 180


def test_unsharp_mask_respects_threshold() -> None:
    pixels = np.full((32, 32, 3), 100, dtype=np.uint8)
    pixels[:, 16:, :] = 104
    source = Image.fromarray(pixels, mode="RGB")

    unchanged = unsharp_mask(source, radius=2.0, percent=200, threshold=5)
    sharpened = unsharp_mask(source, radius=2.0, percent=200, threshold=1)

    np.testing.assert_array_equal(np.asarray(unchanged), pixels)
    assert not np.array_equal(np.asarray(sharpened), pixels)


def test_unsharp_mask_zero_radius_or_percent_returns_identical_image() -> None:
    source = _sample_rgb_image(32, 24)

    np.testing.assert_array_equal(
        np.asarray(unsharp_mask(source, radius=0.0, percent=150, threshold=3)),
        np.asarray(source),
    )
    np.testing.assert_array_equal(
        np.asarray(unsharp_mask(source, radius=2.0, percent=0, threshold=3)),
        np.asarray(source),
    )


def test_unsharp_mask_preserves_alpha_channel() -> None:
    pixels = np.zeros((32, 32, 4), dtype=np.uint8)
    pixels[:, :16, :3] = 70
    pixels[:, 16:, :3] = 190
    pixels[:, :16, 3] = 80
    pixels[:, 16:, 3] = 240
    source = Image.fromarray(pixels, mode="RGBA")

    sharpened = np.asarray(unsharp_mask(source, radius=2.0, percent=150, threshold=3))

    np.testing.assert_array_equal(sharpened[:, :, 3], pixels[:, :, 3])
    assert not np.array_equal(sharpened[:, :, :3], pixels[:, :, :3])


def test_unsharp_mask_matches_reference_closely() -> None:
    source = _sample_rgb_image(64, 48)

    manual = unsharp_mask(source, radius=2.0, percent=150, threshold=3)
    reference = source.filter(ImageFilter.UnsharpMask(radius=2.0, percent=150, threshold=3))

    mae = float(
        np.mean(
            np.abs(np.asarray(manual, dtype=np.float64) - np.asarray(reference, dtype=np.float64))
        )
    )
    assert mae < 1.5


@pytest.mark.parametrize(
    ("radius", "percent", "threshold", "expected_message"),
    [
        (-1.0, 150, 3, "Promień"),
        (2.0, -10, 3, "Siła"),
        (2.0, 150, -1, "Próg"),
    ],
)
def test_unsharp_mask_rejects_negative_arguments(
    radius: float,
    percent: int,
    threshold: int,
    expected_message: str,
) -> None:
    source = Image.new("RGB", (8, 8))

    with pytest.raises(ValueError, match=expected_message):
        unsharp_mask(source, radius=radius, percent=percent, threshold=threshold)
