"""Unified entrypoint for the desktop GUI and frozen extraction worker."""

from collections.abc import Sequence
from io import BytesIO
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

# Minimal valid ONNX Identity model (IR v8, opset 17, float32 [1, 3, H, W] -> [1, 3, H, W]).
_ONNX_IDENTITY_MODEL_BYTES = (
    b'\x08\x08:f\n\x19\n\x05input\x12\x06output"\x08Identity\x12\x0eidentity_check'
    b"Z\x1b\n\x05input\x12\x12\n\x10\x08\x01\x12\x0c\n\x02\x08\x01\n\x02\x08\x03\n\x00\n\x00"
    b"b\x1c\n\x06output\x12\x12\n\x10\x08\x01\x12\x0c\n\x02\x08\x01\n\x02\x08\x03\n\x00\n\x00"
    b"B\x04\n\x00\x10\x11"
)


def _ensure_stdio_streams() -> None:
    """Provide writable streams in windowed binaries where stdout/stderr are None."""
    if sys.stderr is None:
        sys.stderr = Path(os.devnull).open("w", encoding="utf-8")  # noqa: SIM115
    if sys.stdout is None:
        sys.stdout = Path(os.devnull).open("w", encoding="utf-8")  # noqa: SIM115


def _run_self_check() -> int:
    """Verify bundled assets, Qt UI startup, ONNX Runtime session, and worker IPC."""
    if (
        sys.platform != "win32"
        and not os.environ.get("DISPLAY")
        and not os.environ.get("WAYLAND_DISPLAY")
    ):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    import numpy as np
    import onnxruntime as ort
    from PIL import Image
    import pymupdf as fitz
    from PySide6.QtCore import QEventLoop, QTimer
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication

    from pdf_asset_extractor.config import LOGO_PATH, PDF_ICON_PATH, apply_default_config
    from pdf_asset_extractor.onnx_upscaler import detect_default_accelerator
    from pdf_asset_extractor.ui.main_window import MainWindow
    from pdf_asset_extractor.ui.pdf_extraction import PDFExtractionJob

    for asset_path in (LOGO_PATH, PDF_ICON_PATH):
        if not asset_path.is_file():
            raise RuntimeError(f"Missing bundled asset: {asset_path}")

    providers = list(ort.get_available_providers())
    if (
        os.environ.get("PDF_ASSET_EXTRACTOR_REQUIRE_DIRECTML") == "1"
        and "DmlExecutionProvider" not in providers
    ):
        raise RuntimeError(
            f"DirectML provider missing in ONNX Runtime. Available providers: {providers}"
        )

    session = ort.InferenceSession(_ONNX_IDENTITY_MODEL_BYTES, providers=["CPUExecutionProvider"])
    sample_tensor = np.full((1, 3, 16, 16), 0.5, dtype=np.float32)
    output_tensor = np.asarray(session.run(None, {"input": sample_tensor})[0], dtype=np.float32)
    if output_tensor.shape != sample_tensor.shape or not np.allclose(output_tensor, sample_tensor):
        raise RuntimeError(f"Unexpected ONNX self-check output shape: {output_tensor.shape}")

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
                "onnxruntime": ort.__version__,
                "providers": providers,
                "accelerator": detect_default_accelerator(),
                "extracted_images": len(images),
            }
        ),
        flush=True,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch between the extraction worker CLI, self-check, and desktop GUI."""
    _ensure_stdio_streams()
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
