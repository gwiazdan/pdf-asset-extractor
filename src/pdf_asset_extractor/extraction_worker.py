"""Run PDF extraction in a separate process for the desktop application."""

import argparse
import json
from pathlib import Path
import sys

from pdf_asset_extractor.extractor import PDFExtractionError, extract_images_with_thumbnails


def main() -> int:
    """Extract a PDF and print a JSON manifest for the parent process."""
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    arguments = parser.parse_args()

    try:
        images = extract_images_with_thumbnails(
            arguments.pdf_path,
            arguments.output_dir,
            thumbnail_size=256,
        )
    except PDFExtractionError as error:
        print(str(error), file=sys.stderr)
        return 1

    manifest = [
        {
            "page_number": image.page_number,
            "image_index": image.image_index,
            "width": image.width,
            "height": image.height,
            "extension": image.extension,
            "data_file": image.path.name,
            "thumbnail_file": image.thumbnail_path.name,
        }
        for image in images
        if image.path is not None and image.thumbnail_path is not None
    ]
    print(json.dumps(manifest), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
