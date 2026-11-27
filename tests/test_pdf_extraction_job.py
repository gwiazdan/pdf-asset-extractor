from io import BytesIO
import json
import os
from pathlib import Path
import sys

from PIL import Image
import pymupdf as fitz
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication
import pytest

from pdf_asset_extractor.__main__ import _ensure_stdio_streams
from pdf_asset_extractor.__main__ import main as app_main
from pdf_asset_extractor.config import _resolve_base_dir
from pdf_asset_extractor.extractor import ExtractedImage
from pdf_asset_extractor.image_enhancer import EnhancementMethod, EnhancementOptions
from pdf_asset_extractor.ui.export_job import ExportJob
from pdf_asset_extractor.ui.pdf_extraction import PDFExtractionJob, _worker_command


@pytest.fixture(scope="module", autouse=True)
def qt_application() -> QApplication:
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    existing = QApplication.instance()
    if isinstance(existing, QApplication):
        return existing
    return QApplication([])


def _create_pdf(path: Path, *, size: tuple[int, int] = (16, 12)) -> None:
    image_data = BytesIO()
    Image.new("RGB", size, color="red").save(image_data, format="PNG")
    with fitz.open() as document:
        page = document.new_page(width=size[0], height=size[1])
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


def test_extraction_job_preserves_full_resolution_png_separately_from_thumbnail(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "large-png.pdf"
    _create_pdf(pdf_path, size=(600, 400))

    outcome = _run_job(PDFExtractionJob(pdf_path))

    assert "error" not in outcome
    images = outcome["images"]
    assert isinstance(images, list)
    assert len(images) == 1
    with Image.open(BytesIO(images[0].data)) as full_image:
        assert full_image.size == (600, 400)
    with Image.open(BytesIO(images[0].thumbnail)) as thumb_image:
        assert max(thumb_image.size) <= 256


def test_extraction_job_kills_worker_on_timeout(tmp_path: Path) -> None:
    pdf_path = tmp_path / "image.pdf"
    _create_pdf(pdf_path)
    job = PDFExtractionJob(pdf_path, timeout_seconds=30)

    outcome = _run_job(job, force_timeout=True)

    assert "images" not in outcome
    assert "limit 30 sekund" in str(outcome["error"])
    assert job._process.state() == job._process.ProcessState.NotRunning


def test_worker_command_switches_for_frozen_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf_path = tmp_path / "doc.pdf"
    output_dir = tmp_path / "out"

    monkeypatch.delattr(sys, "frozen", raising=False)
    program, args = _worker_command(pdf_path, output_dir)
    assert program == sys.executable
    assert args == [
        "-m",
        "pdf_asset_extractor.extraction_worker",
        str(pdf_path),
        str(output_dir),
    ]

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    frozen_program, frozen_args = _worker_command(pdf_path, output_dir)
    assert frozen_program == sys.executable
    assert frozen_args == ["--extraction-worker", str(pdf_path), str(output_dir)]


def test_main_entrypoint_dispatches_extraction_worker(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pdf_path = tmp_path / "image.pdf"
    output_dir = tmp_path / "worker-out"
    _create_pdf(pdf_path)

    exit_code = app_main(["--extraction-worker", str(pdf_path), str(output_dir)])

    assert exit_code == 0
    manifest = json.loads(capsys.readouterr().out)
    assert len(manifest) == 1
    assert (manifest[0]["width"], manifest[0]["height"]) == (16, 12)
    assert (output_dir / manifest[0]["data_file"]).is_file()


def test_resolve_base_dir_uses_meipass_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert _resolve_base_dir() == tmp_path

    monkeypatch.delattr(sys, "_MEIPASS", raising=False)
    assert (_resolve_base_dir() / "assets" / "logo.svg").is_file()


def test_ensure_stdio_streams_replaces_only_none_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)

    _ensure_stdio_streams()
    new_stdout = sys.stdout
    new_stderr = sys.stderr
    try:
        assert new_stdout is not None
        assert new_stderr is not None
        new_stdout.write("ok")
        new_stderr.write("ok")
        _ensure_stdio_streams()
        assert sys.stdout is new_stdout
        assert sys.stderr is new_stderr
    finally:
        if new_stdout is not None:
            new_stdout.close()
        if new_stderr is not None:
            new_stderr.close()


def test_export_job_saves_images_asynchronously_with_progress(tmp_path: Path) -> None:
    image_buffer = BytesIO()
    Image.new("RGB", (32, 24), color=(50, 120, 200)).save(image_buffer, format="PNG")
    raw_bytes = image_buffer.getvalue()
    extracted = ExtractedImage(
        page_number=1,
        image_index=1,
        width=32,
        height=24,
        extension="png",
        data=raw_bytes,
        thumbnail=raw_bytes,
    )

    event_loop = QEventLoop()
    outcome: dict[str, object] = {}
    statuses: list[str] = []
    job = ExportJob(
        [extracted],
        tmp_path,
        "png",
        EnhancementOptions(method=EnhancementMethod.LANCZOS, scale=2.0),
    )

    def _on_completed(count: int, folder: str) -> None:
        outcome["completed"] = (count, folder)
        event_loop.quit()

    def _on_failed(message: str) -> None:
        outcome["failed"] = message
        event_loop.quit()

    job.status_changed.connect(statuses.append)
    job.completed.connect(_on_completed)
    job.failed.connect(_on_failed)

    safety_timer = QTimer()
    safety_timer.setSingleShot(True)
    safety_timer.timeout.connect(event_loop.quit)
    job.start()
    safety_timer.start(10_000)
    event_loop.exec()
    safety_timer.stop()
    job.wait(5000)
    job.deleteLater()

    assert "failed" not in outcome
    assert outcome["completed"] == (1, str(tmp_path))
    assert statuses
    saved_file = tmp_path / "page-0001-image-0001.png"
    assert saved_file.is_file()
    with Image.open(saved_file) as exported:
        assert exported.size == (64, 48)
