"""Run PDF extraction without blocking the GUI thread."""

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from PySide6.QtCore import QObject, QProcess, QTimer, Signal

from pdf_asset_extractor.extractor import ExtractedImage

PDF_EXTRACTION_TIMEOUT_SECONDS = 60


def _worker_command(pdf_path: Path, output_dir: Path) -> tuple[str, list[str]]:
    """Return the executable and arguments for the extraction subprocess."""
    if getattr(sys, "frozen", False):
        return sys.executable, ["--extraction-worker", str(pdf_path), str(output_dir)]
    return (
        sys.executable,
        [
            "-m",
            "pdf_asset_extractor.extraction_worker",
            str(pdf_path),
            str(output_dir),
        ],
    )


class PDFExtractionJob(QObject):
    """Extract PDF images in a subprocess with a hard timeout."""

    completed = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        pdf_path: Path,
        *,
        timeout_seconds: int = PDF_EXTRACTION_TIMEOUT_SECONDS,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")

        self.pdf_path = pdf_path
        self.timeout_seconds = timeout_seconds
        self._temporary_directory = TemporaryDirectory(prefix="pdf-asset-extraction-")
        self._output_dir = Path(self._temporary_directory.name)
        self._process = QProcess(self)
        self._process.finished.connect(self._handle_finished)
        self._process.errorOccurred.connect(self._handle_process_error)
        self._timeout_timer = QTimer(self)
        self._timeout_timer.setSingleShot(True)
        self._timeout_timer.timeout.connect(self._handle_timeout)
        self._timed_out = False
        self._cancelled = False
        self._finished = False

    def start(self) -> None:
        """Start the extraction worker and its timeout."""
        program, arguments = _worker_command(self.pdf_path, self._output_dir)
        self._process.start(program, arguments)
        self._timeout_timer.start(self.timeout_seconds * 1000)

    def cancel(self) -> None:
        """Terminate the worker when another PDF is selected."""
        self._cancelled = True
        self._timeout_timer.stop()
        if self._process.state() == QProcess.ProcessState.NotRunning:
            self._finish_with_error("Odczyt pliku PDF został anulowany.")
            return
        self._process.kill()

    def _handle_timeout(self) -> None:
        self._timed_out = True
        self._process.kill()

    def _handle_process_error(self, error: QProcess.ProcessError) -> None:
        if error is QProcess.ProcessError.FailedToStart:
            self._finish_with_error("Nie udało się uruchomić procesu odczytu PDF.")

    def _handle_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        self._timeout_timer.stop()
        if self._timed_out:
            self._finish_with_error(
                f"Odczyt pliku PDF przekroczył limit {self.timeout_seconds} sekund."
            )
            return
        if self._cancelled:
            self._finish_with_error("Odczyt pliku PDF został anulowany.")
            return
        if exit_status is not QProcess.ExitStatus.NormalExit or exit_code != 0:
            message = (
                bytes(self._process.readAllStandardError().data())
                .decode("utf-8", errors="replace")
                .strip()
            )
            self._finish_with_error(message or "Nie udało się odczytać pliku PDF.")
            return

        try:
            manifest = json.loads(bytes(self._process.readAllStandardOutput().data()))
            images = self._load_images(manifest)
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            self._finish_with_error(f"Nie można odczytać wyników ekstrakcji PDF: {error}")
            return

        self._finish(images)

    def _load_images(self, manifest: object) -> list[ExtractedImage]:
        """Load extracted image data and thumbnails from the worker directory."""
        if not isinstance(manifest, list):
            raise ValueError("nieprawidłowy manifest obrazów")

        images: list[ExtractedImage] = []
        for item in manifest:
            if not isinstance(item, dict):
                raise ValueError("nieprawidłowy wpis obrazu")
            image_path = self._output_dir / item["data_file"]
            thumbnail_path = self._output_dir / item["thumbnail_file"]
            images.append(
                ExtractedImage(
                    page_number=item["page_number"],
                    image_index=item["image_index"],
                    width=item["width"],
                    height=item["height"],
                    extension=item["extension"],
                    data=image_path.read_bytes(),
                    thumbnail=thumbnail_path.read_bytes(),
                )
            )
        return images

    def _finish_with_error(self, message: str) -> None:
        self._finish(message)

    def _finish(self, result: list[ExtractedImage] | str) -> None:
        """Emit a terminal signal once and always clean up the worker directory."""
        if self._finished:
            return
        self._finished = True
        self._timeout_timer.stop()
        self._temporary_directory.cleanup()
        if isinstance(result, str):
            self.failed.emit(result)
            return
        self.completed.emit(result)
