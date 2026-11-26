from io import BytesIO
import os
from pathlib import Path

from PIL import Image
import pymupdf as fitz
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication
import pytest

from pdf_asset_extractor.ui.pdf_extraction import PDFExtractionJob


@pytest.fixture(scope="module", autouse=True)
def qt_application() -> QApplication:
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    existing = QApplication.instance()
    if isinstance(existing, QApplication):
        return existing
    return QApplication([])


def _create_pdf(path: Path) -> None:
    image_data = BytesIO()
    Image.new("RGB", (16, 12), color="red").save(image_data, format="PNG")
    with fitz.open() as document:
        page = document.new_page(width=16, height=12)
        page.insert_image(page.rect, stream=image_data.getvalue())
        document.save(path)


def _run_job(job: PDFExtractionJob, *, force_timeout: bool = False) -> dict[str, object]:
    event_loop = QEventLoop()
    outcome: dict[str, object] = {}

    def handle_completed(images: object) -> None:
        outcome["images"] = images
        event_loop.quit()

    def handle_failed(message: str) -> None:
        outcome["error"] = message
        event_loop.quit()

    job.completed.connect(handle_completed)
    job.failed.connect(handle_failed)
    if force_timeout:
        QTimer.singleShot(1, job._handle_timeout)

    safety_timer = QTimer()
    safety_timer.setSingleShot(True)
    safety_timer.timeout.connect(event_loop.quit)
    job.start()
    safety_timer.start(10_000)
    event_loop.exec()
    safety_timer.stop()
    job.deleteLater()
    return outcome


def test_extraction_job_returns_images(tmp_path: Path) -> None:
    pdf_path = tmp_path / "image.pdf"
    _create_pdf(pdf_path)

    outcome = _run_job(PDFExtractionJob(pdf_path))

    assert "error" not in outcome
    images = outcome["images"]
    assert isinstance(images, list)
    assert len(images) == 1
    assert images[0].width == 16
    assert images[0].height == 12


def test_extraction_job_kills_worker_on_timeout(tmp_path: Path) -> None:
    pdf_path = tmp_path / "image.pdf"
    _create_pdf(pdf_path)
    job = PDFExtractionJob(pdf_path, timeout_seconds=30)

    outcome = _run_job(job, force_timeout=True)

    assert "images" not in outcome
    assert "limit 30 sekund" in str(outcome["error"])
    assert job._process.state() == job._process.ProcessState.NotRunning
