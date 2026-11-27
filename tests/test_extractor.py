"""Unit tests for PDF image extraction, format encoding, and worker CLI."""

from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image
import pymupdf as fitz
import pytest

from pdf_asset_extractor.extractor import (
    ExtractedImage,
    ImageConversionError,
    PDFExtractionError,
    encode_image,
    extract_images_with_thumbnails,
)


def _png_bytes(
    size: tuple[int, int],
    *,
    mode: str = "RGB",
    color: int | tuple[int, ...] = (200, 40, 20),
) -> bytes:
    buffer = BytesIO()
    Image.new(mode, size, color=color).save(buffer, format="PNG")
    return buffer.getvalue()


def _create_pdf_with_images(
    pdf_path: Path,
    images: list[tuple[tuple[int, int], bytes]],
    *,
    password: str | None = None,
) -> None:
    with fitz.open() as document:
        for (width, height), stream in images:
            page = document.new_page(width=width, height=height)
            page.insert_image(page.rect, stream=stream)
        if password is not None:
            document.save(
                pdf_path,
                encryption=fitz.PDF_ENCRYPT_AES_256,
                user_pw=password,
                owner_pw=password,
            )
        else:
            document.save(pdf_path)


def test_extract_images_across_pages_and_write_outputs(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    output_dir = tmp_path / "extracted"
    _create_pdf_with_images(
        pdf_path,
        [
            ((64, 48), _png_bytes((64, 48), color=(255, 0, 0))),
            ((256, 128), _png_bytes((256, 128), color=(0, 255, 0))),
        ],
    )

    extracted = extract_images_with_thumbnails(pdf_path, output_dir, thumbnail_size=64)

    assert len(extracted) == 2
    assert [(img.page_number, img.image_index) for img in extracted] == [(1, 1), (2, 1)]
    assert (extracted[0].width, extracted[0].height) == (64, 48)
    assert (extracted[1].width, extracted[1].height) == (256, 128)

    for image in extracted:
        assert image.path is not None
        assert image.path.is_file()
        assert image.thumbnail_path is not None
        assert image.thumbnail_path.is_file()
        assert image.path != image.thumbnail_path
        with Image.open(image.path) as saved_image:
            assert saved_image.size == (image.width, image.height)
        with Image.open(image.thumbnail_path) as saved_thumb:
            assert max(saved_thumb.size) <= 64
        with Image.open(BytesIO(image.thumbnail)) as thumb:
            assert max(thumb.size) <= 64


def test_extract_images_rejects_non_positive_thumbnail_size(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="thumbnail_size"):
        extract_images_with_thumbnails(tmp_path / "missing.pdf", thumbnail_size=0)


def test_extract_images_handles_password_protected_pdf(tmp_path: Path) -> None:
    pdf_path = tmp_path / "locked.pdf"
    _create_pdf_with_images(
        pdf_path,
        [((24, 24), _png_bytes((24, 24), color=(10, 20, 30)))],
        password="secret",
    )

    with pytest.raises(PDFExtractionError, match="hasłem"):
        extract_images_with_thumbnails(pdf_path)

    with pytest.raises(PDFExtractionError, match="hasłem"):
        extract_images_with_thumbnails(pdf_path, password="wrong")

    extracted = extract_images_with_thumbnails(pdf_path, password="secret")
    assert len(extracted) == 1
    assert (extracted[0].width, extracted[0].height) == (24, 24)


def test_extract_images_raises_on_missing_or_corrupt_pdf(tmp_path: Path) -> None:
    with pytest.raises(PDFExtractionError, match="Nie można otworzyć"):
        extract_images_with_thumbnails(tmp_path / "nonexistent.pdf")

    corrupt_pdf = tmp_path / "corrupt.pdf"
    corrupt_pdf.write_bytes(b"not a valid pdf")
    with pytest.raises(PDFExtractionError, match="Nie można otworzyć"):
        extract_images_with_thumbnails(corrupt_pdf)


def test_encode_image_formats_and_colorspaces() -> None:
    rgba_bytes = _png_bytes((16, 12), mode="RGBA", color=(120, 180, 240, 128))
    extracted = ExtractedImage(
        page_number=1,
        image_index=1,
        width=16,
        height=12,
        extension="png",
        data=rgba_bytes,
        thumbnail=rgba_bytes,
    )

    original_data, original_ext = encode_image(extracted, "original")
    assert original_data == rgba_bytes
    assert original_ext == "png"

    png_data, png_ext = encode_image(extracted, "png")
    assert png_ext == "png"
    with Image.open(BytesIO(png_data)) as decoded_png:
        assert decoded_png.size == (16, 12)

    jpg_data, jpg_ext = encode_image(extracted, "jpg")
    assert jpg_ext == "jpg"
    with Image.open(BytesIO(jpg_data)) as decoded_jpg:
        assert decoded_jpg.mode == "RGB"
        assert decoded_jpg.size == (16, 12)

    # Verify CMYK inputs can be encoded to both PNG and JPEG without failing.
    cmyk_buffer = BytesIO()
    Image.new("CMYK", (12, 10), color=(20, 40, 60, 10)).save(cmyk_buffer, format="JPEG")
    cmyk_extracted = ExtractedImage(
        page_number=1,
        image_index=2,
        width=12,
        height=10,
        extension="jpg",
        data=cmyk_buffer.getvalue(),
        thumbnail=b"",
    )
    cmyk_png, _ = encode_image(cmyk_extracted, "png")
    with Image.open(BytesIO(cmyk_png)) as decoded_cmyk_png:
        assert decoded_cmyk_png.mode == "RGB"
        assert decoded_cmyk_png.size == (12, 10)


def test_encode_image_rejects_invalid_format_and_bad_bytes() -> None:
    valid_bytes = _png_bytes((8, 8))
    extracted = ExtractedImage(1, 1, 8, 8, "png", valid_bytes, valid_bytes)

    with pytest.raises(ValueError, match="Unsupported output format"):
        encode_image(extracted, "webp")

    broken = ExtractedImage(1, 1, 8, 8, "png", b"corrupt", b"")
    with pytest.raises(ImageConversionError, match="Nie można przekonwertować"):
        encode_image(broken, "png")


def test_extraction_worker_cli_outputs_manifest_and_errors(tmp_path: Path) -> None:
    pdf_path = tmp_path / "worker.pdf"
    output_dir = tmp_path / "worker-out"
    _create_pdf_with_images(pdf_path, [((20, 16), _png_bytes((20, 16), color=(0, 128, 255)))])

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pdf_asset_extractor.extraction_worker",
            str(pdf_path),
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    manifest = json.loads(completed.stdout)
    assert len(manifest) == 1
    assert manifest[0]["width"] == 20
    assert manifest[0]["height"] == 16
    assert (output_dir / manifest[0]["data_file"]).is_file()
    assert (output_dir / manifest[0]["thumbnail_file"]).is_file()

    failed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pdf_asset_extractor.extraction_worker",
            str(tmp_path / "missing.pdf"),
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert failed.returncode == 1
    assert "Nie można otworzyć" in failed.stderr
