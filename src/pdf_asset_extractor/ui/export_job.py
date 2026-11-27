"""Asynchronous image export worker with live progress, cancellation, and error logging."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
import threading
import traceback

from PySide6.QtCore import QObject, QThread, Signal, Slot

from pdf_asset_extractor.extractor import ExtractedImage, ImageConversionError, encode_image
from pdf_asset_extractor.image_enhancer import (
    EnhancementCancelledError,
    EnhancementError,
    EnhancementMethod,
    EnhancementOptions,
    enhance_image,
)
from pdf_asset_extractor.onnx_upscaler import default_app_cache_dir


def app_log_path() -> Path:
    """Return the path to the diagnostic log file."""
    return default_app_cache_dir() / "app.log"


def _log_exception(context: str, error: BaseException) -> Path | None:
    """Append an error traceback to the local diagnostic log without raising."""
    log_file = app_log_path()
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().isoformat(timespec="seconds")
        details = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        with log_file.open("a", encoding="utf-8") as handle:
            handle.write(f"[{timestamp}] {context}\n{details}\n")
        return log_file
    except OSError:
        return None


class _ExportWorker(QObject):
    """Worker executed inside a dedicated QThread."""

    status_changed = Signal(str)
    progress_changed = Signal(int, int)
    completed = Signal(int, str)
    cancelled = Signal(int, str)
    failed = Signal(str)
    finished = Signal()

    def __init__(
        self,
        images: Sequence[ExtractedImage],
        destination: Path,
        output_format: str,
        enhancement_options: EnhancementOptions,
        cancel_event: threading.Event,
    ) -> None:
        super().__init__()
        self._images = list(images)
        self._destination = destination
        self._output_format = output_format
        self._options = enhancement_options
        self._cancel_event = cancel_event

    @Slot()
    def run(self) -> None:
        saved_count = 0
        total_images = len(self._images)
        try:
            for index, image in enumerate(self._images, start=1):
                if self._cancel_event.is_set():
                    self.cancelled.emit(saved_count, str(self._destination))
                    return

                self.status_changed.emit(f"Przetwarzanie obrazu {index}/{total_images}...")
                if self._options.method is not EnhancementMethod.SUPER_IMAGE:
                    self.progress_changed.emit(index - 1, total_images)

                def _on_progress(
                    message: str,
                    current: int,
                    maximum: int,
                    *,
                    img_idx: int = index,
                ) -> None:
                    prefix = (
                        f"Obraz {img_idx}/{total_images}: "
                        if total_images > 1 and not message.startswith("Pobieranie")
                        else ""
                    )
                    self.status_changed.emit(f"{prefix}{message}")
                    self.progress_changed.emit(current, maximum)

                enhanced_data = enhance_image(
                    image.data,
                    self._options,
                    progress_callback=_on_progress,
                    cancel_check=self._cancel_event.is_set,
                )

                if self._cancel_event.is_set():
                    self.cancelled.emit(saved_count, str(self._destination))
                    return

                if self._options.method is EnhancementMethod.NONE:
                    image_data, extension = encode_image(image, self._output_format)
                else:
                    enhanced_image = ExtractedImage(
                        page_number=image.page_number,
                        image_index=image.image_index,
                        width=image.width,
                        height=image.height,
                        extension="png",
                        data=enhanced_data,
                        thumbnail=image.thumbnail,
                    )
                    image_data, extension = encode_image(enhanced_image, self._output_format)

                filename = (
                    f"page-{image.page_number:04d}-image-{image.image_index:04d}.{extension}"
                )
                self.status_changed.emit(f"Zapisywanie {filename} ({index}/{total_images})...")
                (self._destination / filename).write_bytes(image_data)
                saved_count += 1
                self.progress_changed.emit(index, total_images)

            self.completed.emit(saved_count, str(self._destination))
        except EnhancementCancelledError:
            self.cancelled.emit(saved_count, str(self._destination))
        except (EnhancementError, ImageConversionError, OSError) as error:
            _log_exception("Export failed", error)
            self.failed.emit(str(error))
        except Exception as error:
            log_file = _log_exception("Unexpected export error", error)
            suffix = f"\n\nSzczegóły zapisano w logu: {log_file}" if log_file is not None else ""
            self.failed.emit(f"Nieoczekiwany błąd eksportu: {error}{suffix}")
        finally:
            self.finished.emit()


class ExportJob(QObject):
    """Run batch image enhancement and file export off the Qt GUI thread."""

    status_changed = Signal(str)
    progress_changed = Signal(int, int)
    completed = Signal(int, str)
    cancelled = Signal(int, str)
    failed = Signal(str)

    def __init__(
        self,
        images: Sequence[ExtractedImage],
        destination: Path,
        output_format: str,
        enhancement_options: EnhancementOptions,
        *,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._cancel_event = threading.Event()
        self._thread = QThread(self)
        self._worker = _ExportWorker(
            images,
            destination,
            output_format,
            enhancement_options,
            self._cancel_event,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.status_changed.connect(self.status_changed)
        self._worker.progress_changed.connect(self.progress_changed)
        self._worker.completed.connect(self.completed)
        self._worker.cancelled.connect(self.cancelled)
        self._worker.failed.connect(self.failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)

    def start(self) -> None:
        """Start the background export thread."""
        self._thread.start()

    def cancel(self) -> None:
        """Request cooperative cancellation of the running export."""
        self._cancel_event.set()
        self.status_changed.emit("Anulowanie eksportu...")

    def wait(self, timeout_ms: int = 5000) -> bool:
        """Wait for the worker thread to stop."""
        return self._thread.wait(timeout_ms)
