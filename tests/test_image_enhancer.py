from io import BytesIO

import numpy as np
from PIL import Image
import pytest
import torch

import pdf_asset_extractor.image_enhancer as image_enhancer
from pdf_asset_extractor.image_enhancer import (
    EnhancementMethod,
    EnhancementOptions,
    ResizeMode,
    _resolve_device,
    enhance_image,
    enhance_images,
)


class NearestUpscaler:
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.interpolate(images, scale_factor=2, mode="nearest")


def _encode_image(color: tuple[int, int, int]) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (8, 8), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_auto_device_falls_back_to_cpu_without_cuda() -> None:
    assert _resolve_device("auto", cuda_available=False) == "cpu"


@pytest.mark.parametrize(
    "device",
    ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",),
)
def test_super_image_batch_matches_single_image_outputs(monkeypatch, device: str) -> None:
    monkeypatch.setattr(
        image_enhancer,
        "_load_super_image_model",
        lambda model_name, scale, device: NearestUpscaler(),
    )
    options = EnhancementOptions(
        method=EnhancementMethod.SUPER_IMAGE,
        scale=2,
        model_name="edsr",
        device=device,
    )
    image_data = [_encode_image((255, 0, 0)), _encode_image((0, 255, 0))]

    batch_results = enhance_images(image_data, options)
    single_results = [enhance_image(image, options) for image in image_data]

    for batch_result, single_result in zip(batch_results, single_results, strict=True):
        with Image.open(BytesIO(batch_result)) as batch_image:
            batch_pixels = np.asarray(batch_image)
        with Image.open(BytesIO(single_result)) as single_image:
            single_pixels = np.asarray(single_image)
        np.testing.assert_array_equal(batch_pixels, single_pixels)


def test_lanczos_enhancement_resizes_without_pil_resize(monkeypatch) -> None:
    def _forbid_pil_resize(*args, **kwargs):
        raise AssertionError("PIL Image.resize should not be called")

    monkeypatch.setattr(Image.Image, "resize", _forbid_pil_resize)

    image_bytes = _encode_image((120, 60, 200))
    options = EnhancementOptions(
        method=EnhancementMethod.LANCZOS,
        resize_mode=ResizeMode.WIDTH,
        target_width=24,
    )

    enhanced_bytes = enhance_image(image_bytes, options)
    with Image.open(BytesIO(enhanced_bytes)) as enhanced:
        assert enhanced.size == (24, 24)
        assert np.all(np.asarray(enhanced) == np.array([120, 60, 200], dtype=np.uint8))


def test_unsharp_mask_enhancement_sharpens_without_pil_filter(monkeypatch) -> None:
    def _forbid_pil_filter(*args, **kwargs):
        raise AssertionError("PIL Image.filter should not be called")

    monkeypatch.setattr(Image.Image, "filter", _forbid_pil_filter)

    pixels = np.full((24, 24, 3), 60, dtype=np.uint8)
    pixels[:, 12:, :] = 200
    buffer = BytesIO()
    Image.fromarray(pixels, mode="RGB").save(buffer, format="PNG")

    options = EnhancementOptions(
        method=EnhancementMethod.UNSHARP_MASK,
        unsharp_radius=2.0,
        unsharp_percent=150,
        unsharp_threshold=3,
    )
    enhanced_bytes = enhance_image(buffer.getvalue(), options)
    with Image.open(BytesIO(enhanced_bytes)) as enhanced:
        result = np.asarray(enhanced)
        assert enhanced.size == (24, 24)
        assert int(result[12, 11, 0]) < 60
        assert int(result[12, 12, 0]) > 200


def test_super_image_width_mode_uses_manual_lanczos(monkeypatch) -> None:
    def _forbid_pil_resize(*args, **kwargs):
        raise AssertionError("PIL Image.resize should not be called")

    monkeypatch.setattr(Image.Image, "resize", _forbid_pil_resize)
    monkeypatch.setattr(
        image_enhancer,
        "_load_super_image_model",
        lambda model_name, scale, device: NearestUpscaler(),
    )

    options = EnhancementOptions(
        method=EnhancementMethod.SUPER_IMAGE,
        resize_mode=ResizeMode.WIDTH,
        scale=2.0,
        target_width=20,
        model_name="edsr",
        device="cpu",
    )
    enhanced_bytes = enhance_image(_encode_image((255, 0, 0)), options)
    with Image.open(BytesIO(enhanced_bytes)) as enhanced:
        assert enhanced.size == (20, 20)
