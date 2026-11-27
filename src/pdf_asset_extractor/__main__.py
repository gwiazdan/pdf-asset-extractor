"""Unified entrypoint for the desktop GUI and frozen extraction worker."""

from collections.abc import Sequence
from io import BytesIO
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory


def _run_self_check() -> int:
    """Verify bundled assets, Qt UI startup, ML imports, and worker subprocess IPC."""
    if (
        sys.platform != "win32"
        and not os.environ.get("DISPLAY")
        and not os.environ.get("WAYLAND_DISPLAY")
    ):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    import cv2  # noqa: F401
    from PIL import Image
    import pymupdf as fitz
    from PySide6.QtCore import QEventLoop, QTimer
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication
    import super_image
    import torch

    from pdf_asset_extractor.config import LOGO_PATH, PDF_ICON_PATH, apply_default_config
    from pdf_asset_extractor.image_enhancer import SUPER_IMAGE_MODELS
    from pdf_asset_extractor.ui.main_window import MainWindow
    from pdf_asset_extractor.ui.pdf_extraction import PDFExtractionJob

    for asset_path in (LOGO_PATH, PDF_ICON_PATH):
        if not asset_path.is_file():
            raise RuntimeError(f"Missing bundled asset: {asset_path}")

    for _, model_class_name in SUPER_IMAGE_MODELS:
        if not hasattr(super_image, model_class_name):
            raise RuntimeError(f"Missing super_image model class: {model_class_name}")

    existing_app = QApplication.instance()
    app = existing_app if isinstance(existing_app, QApplication) else QApplication([])
    apply_default_config()

    for asset_path in (LOGO_PATH, PDF_ICON_PATH):
        icon = QIcon(str(asset_path))
        if icon.isNull() or icon.pixmap(32, 32).isNull():
            raise RuntimeError(f"Failed to render SVG icon: {asset_path}")

    window = MainWindow()
    window.close()

    with TemporaryDirectory(prefix="pdf-asset-self-check-") as temp_dir:
        pdf_path = Path(temp_dir) / "sample.pdf"
        image_buffer = BytesIO()
        Image.new("RGB", (512, 384), color="red").save(image_buffer, format="PNG")
        with fitz.open() as document:
            page = document.new_page(width=512, height=384)
            page.insert_image(page.rect, stream=image_buffer.getvalue())
            document.save(pdf_path)

        outcome: dict[str, object] = {}
        event_loop = QEventLoop()
        job = PDFExtractionJob(pdf_path, timeout_seconds=30)

        def handle_completed(images: object) -> None:
            outcome["images"] = images
            event_loop.quit()

        def handle_failed(message: str) -> None:
            outcome["error"] = message
            event_loop.quit()

        job.completed.connect(handle_completed)
        job.failed.connect(handle_failed)

        safety_timer = QTimer()
        safety_timer.setSingleShot(True)
        safety_timer.timeout.connect(event_loop.quit)
        job.start()
        safety_timer.start(30_000)
        event_loop.exec()
        safety_timer.stop()
        job.deleteLater()
        app.processEvents()

    if "error" in outcome:
        raise RuntimeError(f"Extraction worker failed: {outcome['error']}")
    images = outcome.get("images")
    if not isinstance(images, list) or len(images) != 1:
        raise RuntimeError(f"Unexpected extraction output: {outcome!r}")
    first_image = images[0]
    metadata_size = (first_image.width, first_image.height)
    with Image.open(BytesIO(first_image.data)) as extracted_image:
        extracted_size = extracted_image.size
    if metadata_size != (512, 384) or extracted_size != (512, 384):
        raise RuntimeError(
            f"Unexpected extracted image size: metadata={metadata_size}, pixels={extracted_size}"
        )

    print(
        json.dumps(
            {
                "status": "ok",
                "frozen": bool(getattr(sys, "frozen", False)),
                "torch": torch.__version__,
                "extracted_images": len(images),
            }
        ),
        flush=True,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch between the extraction worker CLI, self-check, and desktop GUI."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "--extraction-worker":
        from pdf_asset_extractor.extraction_worker import main as worker_main

        return worker_main(args[1:])
    if args and args[0] == "--self-check":
        return _run_self_check()

    from pdf_asset_extractor.ui.main_window import main as gui_main

    return gui_main()


if __name__ == "__main__":
    raise SystemExit(main())
