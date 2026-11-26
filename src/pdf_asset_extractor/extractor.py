"""Extract embedded PDF images and renderable thumbnails."""

from dataclasses import dataclass
from math import ceil, log2
from pathlib import Path

import pymupdf as fitz


class PDFExtractionError(RuntimeError):
    """Raised when a PDF cannot be opened or an image cannot be extracted."""


class ImageConversionError(RuntimeError):
    """Raised when an extracted image cannot be encoded in the requested format."""


@dataclass(frozen=True, slots=True)
class ExtractedImage:
    """An embedded image together with its PDF metadata and thumbnail."""

    page_number: int
    image_index: int
    width: int
    height: int
    extension: str
    data: bytes
    thumbnail: bytes
    path: Path | None = None
    thumbnail_path: Path | None = None


def _ensure_png_compatible(pixmap: fitz.Pixmap, *, drop_alpha: bool = False) -> fitz.Pixmap:
    """Convert non-grayscale/non-RGB pixmaps (such as CMYK) to RGB for PNG encoding."""
    if drop_alpha and pixmap.alpha:
        return fitz.Pixmap(fitz.csRGB, pixmap)
    if pixmap.colorspace is not None and pixmap.colorspace.n not in {1, 3}:
        return fitz.Pixmap(fitz.csRGB, pixmap)
    return pixmap


def _thumbnail(image_data: bytes, max_size: int) -> bytes:
    """Create a PNG thumbnail from raw image bytes."""
    image = _ensure_png_compatible(fitz.Pixmap(image_data), drop_alpha=True)

    scale = min(max_size / image.width, max_size / image.height, 1)
    if scale < 1:
        # Pixmap.shrink halves each dimension 2**k times, which is much faster than full resample.
        shrink_power = ceil(log2(1 / scale))
        image.shrink(shrink_power)
    return image.tobytes("png")


def _output_paths(
    output_dir: Path, page_number: int, image_index: int, extension: str
) -> tuple[Path, Path]:
    """Return stable paths for one extracted image and its thumbnail."""
    stem = f"page-{page_number:04d}-image-{image_index:04d}"
    return output_dir / f"{stem}.{extension}", output_dir / f"{stem}.png"


def extract_images_with_thumbnails(
    pdf_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    password: str | None = None,
    thumbnail_size: int = 128,
) -> list[ExtractedImage]:
    """Extract every placed image from a PDF and create a PNG thumbnail."""
    if thumbnail_size <= 0:
        raise ValueError("thumbnail_size must be greater than zero")

    source_path = Path(pdf_path)
    destination = Path(output_dir) if output_dir is not None else None
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)

    try:
        document = fitz.open(source_path)
    except (
        OSError,
        ValueError,
        fitz.EmptyFileError,
        fitz.FileDataError,
        fitz.FileNotFoundError,
    ) as error:
        raise PDFExtractionError(f"Nie można otworzyć pliku PDF: {source_path}") from error

    with document:
        if document.needs_pass and (password is None or not document.authenticate(password)):
            raise PDFExtractionError("Plik PDF jest zabezpieczony hasłem.")

        extracted: list[ExtractedImage] = []
        for page_number, page in enumerate(document.pages(), start=1):
            for image_index, image_info in enumerate(page.get_images(), start=1):
                try:
                    image = document.extract_image(image_info[0])
                    image_data = image["image"]
                    thumbnail_data = _thumbnail(image_data, thumbnail_size)
                except (KeyError, ValueError, fitz.FileDataError) as error:
                    raise PDFExtractionError(
                        f"Nie można wyodrębnić obrazu {image_index} na stronie {page_number}."
                    ) from error

                image_path: Path | None = None
                thumbnail_path: Path | None = None
                if destination is not None:
                    image_path, thumbnail_path = _output_paths(
                        destination, page_number, image_index, image["ext"]
                    )
                    image_path.write_bytes(image_data)
                    thumbnail_path.write_bytes(thumbnail_data)

                extracted.append(
                    ExtractedImage(
                        page_number=page_number,
                        image_index=image_index,
                        width=image["width"],
                        height=image["height"],
                        extension=image["ext"],
                        data=image_data,
                        thumbnail=thumbnail_data,
                        path=image_path,
                        thumbnail_path=thumbnail_path,
                    )
                )

    return extracted


def encode_image(image: ExtractedImage, output_format: str) -> tuple[bytes, str]:
    """Encode an extracted image and return its data and file extension."""
    if output_format == "original":
        return image.data, image.extension

    if output_format not in {"png", "jpg"}:
        raise ValueError(f"Unsupported output format: {output_format}")

    try:
        pixmap = fitz.Pixmap(image.data)
        if output_format == "jpg":
            # JPEG does not support alpha or non-RGB colorspaces; drop alpha before RGB conversion.
            if pixmap.alpha:
                pixmap = fitz.Pixmap(pixmap, 0)
            if pixmap.colorspace is None or pixmap.colorspace.n != 3:
                pixmap = fitz.Pixmap(fitz.csRGB, pixmap)
        else:
            pixmap = _ensure_png_compatible(pixmap)
        return pixmap.tobytes(output_format), output_format
    except Exception as error:
        raise ImageConversionError(
            f"Nie można przekonwertować obrazu {image.image_index} "
            f"na stronie {image.page_number} do formatu {output_format.upper()}."
        ) from error
