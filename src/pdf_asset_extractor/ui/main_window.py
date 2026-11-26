"""Main desktop window for browsing PDFs and exporting embedded images."""

import os
from pathlib import Path
import sys
from typing import Final

from PySide6.QtCore import QRect, QSize, Qt
from PySide6.QtGui import QFontMetrics, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFileDialog,
    QFileIconProvider,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    CaptionLabel,
    ComboBox,
    FluentIcon,
    ListWidget,
    MessageBox,
    MSFluentWindow,
    PushButton,
    StrongBodyLabel,
)

from pdf_asset_extractor.config import (
    ACRYLIC_ENABLED,
    LOGO_PATH,
    PDF_TILE_SIZE,
    apply_default_config,
)
from pdf_asset_extractor.extractor import ExtractedImage, ImageConversionError, encode_image
from pdf_asset_extractor.image_enhancer import (
    SUPER_IMAGE_MODELS,
    EnhancementError,
    EnhancementMethod,
    EnhancementOptions,
    ResizeMode,
    enhance_images,
)
from pdf_asset_extractor.ui.file_utils import icon_for_path
from pdf_asset_extractor.ui.folder_picker import FolderPickerWidget
from pdf_asset_extractor.ui.pdf_extraction import PDFExtractionJob

PDF_TILE_TEXT_WIDTH: Final[int] = PDF_TILE_SIZE.width() - 16
ENHANCEMENT_METHODS: Final[tuple[tuple[str, EnhancementMethod], ...]] = (
    ("Bez poprawy", EnhancementMethod.NONE),
    ("Skalowanie Lanczosem", EnhancementMethod.LANCZOS),
    ("Wyostrzanie maską", EnhancementMethod.UNSHARP_MASK),
    ("Super-image", EnhancementMethod.SUPER_IMAGE),
)
OUTPUT_FORMAT_MAP: Final[dict[str, str]] = {
    "Oryginalny": "original",
    "PNG": "png",
    "JPEG": "jpg",
}
DEVICE_MAP: Final[dict[str, str]] = {
    "Auto": "auto",
    "CPU": "cpu",
    "CUDA": "cuda",
}


def _parse_unsharp_preset(preset: str) -> tuple[float, int, int]:
    """Parse a 'radius / percent / threshold' preset label into numeric values."""
    parts = [part.strip() for part in preset.split("/")]
    if len(parts) != 3:
        raise ValueError(f"Invalid unsharp mask preset: {preset}")
    return float(parts[0]), int(parts[1]), int(parts[2])


def _fit_thumbnail(pixmap: QPixmap, size: QSize) -> QPixmap:
    """Fit a thumbnail inside a square canvas without cropping it."""
    scaled = pixmap.scaled(
        size,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
    canvas = QPixmap(size)
    canvas.fill(Qt.GlobalColor.transparent)
    painter = QPainter(canvas)
    x = (size.width() - scaled.width()) // 2
    y = (size.height() - scaled.height()) // 2
    painter.drawPixmap(x, y, scaled)
    painter.end()
    return canvas


def _wrap_pdf_filename(filename: str, font_metrics: QFontMetrics) -> str:
    """Insert display-only line breaks so even unbroken filenames fit a tile."""
    lines: list[str] = []
    current_line = ""
    for character in filename:
        candidate = current_line + character
        if current_line and font_metrics.horizontalAdvance(candidate) > PDF_TILE_TEXT_WIDTH:
            lines.append(current_line)
            current_line = character
        else:
            current_line = candidate
    if current_line:
        lines.append(current_line)
    return "\n".join(lines)


def _pdf_tile_size(
    display_name: str,
    font_metrics: QFontMetrics,
    icon_size: QSize,
) -> QSize:
    """Return a tile size that gives the wrapped filename enough vertical space."""
    text_bounds = font_metrics.boundingRect(
        QRect(0, 0, PDF_TILE_TEXT_WIDTH, 10_000),
        Qt.TextFlag.TextWordWrap,
        display_name,
    )
    text_height = text_bounds.height()
    tile_height = max(PDF_TILE_SIZE.height(), icon_size.height() + text_height + 32)
    return QSize(PDF_TILE_SIZE.width(), tile_height)


class MainWindow(MSFluentWindow):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()
        self.init_window()

    def init_window(self) -> None:
        """Initialize window layout, widgets, and navigation views."""
        self.resize(1024, 768)
        self.setWindowTitle("PDF Asset Extractor")
        self.setWindowIcon(QIcon(str(LOGO_PATH)))
        self.titleBar.setFixedHeight(45)
        self.selected_pdf: Path | None = None
        self.extracted_images: list[ExtractedImage] = []
        self._extraction_job: PDFExtractionJob | None = None

        # Instantiate view widgets upfront so every attribute is initialized cleanly.
        self.folder_label = StrongBodyLabel()
        self.content_list = ListWidget()
        self.selected_pdf_label = StrongBodyLabel("Nie wybrano pliku PDF")
        self.image_list = ListWidget()
        self.enhancement_combo: ComboBox = ComboBox()
        self.enhancement_resize_mode_combo: ComboBox = ComboBox()
        self.enhancement_scale_combo: ComboBox = ComboBox()
        self.enhancement_width_combo: ComboBox = ComboBox()
        self.enhancement_model_combo: ComboBox = ComboBox()
        self.enhancement_device_combo: ComboBox = ComboBox()
        self.enhancement_unsharp_combo: ComboBox = ComboBox()
        self.output_format_combo = ComboBox()
        self.export_button = PushButton("Eksportuj zaznaczone obrazy")

        self._folder_picker = FolderPickerWidget()
        self._folder_picker.setObjectName("folderInterface")
        self._folder_picker.folder_selected.connect(self._show_folder_content)
        self._content_view = self._create_content_view()
        self._content_view.setObjectName("folderContentInterface")
        self._pdf_view = self._create_pdf_view()
        self._pdf_view.setObjectName("pdfInterface")

        self._folder_navigation = self.addSubInterface(
            self._content_view, FluentIcon.FOLDER, "Folder"
        )
        self.stackedWidget.addWidget(self._folder_picker)
        self._pdf_navigation = self.addSubInterface(
            self._pdf_view, FluentIcon.DOCUMENT, "Plik PDF"
        )
        self._pdf_navigation.setEnabled(False)
        self.switchTo(self._folder_picker)

        if ACRYLIC_ENABLED:
            try:
                self.windowEffect.setAcrylicEffect(self.winId())
            except Exception as error:
                print(f"Nie udało się włączyć efektu akrylowego: {error}")

    def _create_content_view(self) -> QWidget:
        view = QWidget()
        layout = QVBoxLayout(view)
        layout.setContentsMargins(24, 24, 24, 24)

        header_layout = QHBoxLayout()
        self.folder_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        choose_button = PushButton(FluentIcon.FOLDER_ADD, "Wybierz inny folder")
        choose_button.clicked.connect(self.open_folder)
        header_layout.addWidget(self.folder_label, 1)
        header_layout.addWidget(choose_button)

        self.content_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.content_list.setIconSize(QSize(64, 64))
        self.content_list.setGridSize(QSize())
        self.content_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.content_list.setMovement(QListWidget.Movement.Static)
        self.content_list.setSpacing(12)
        self.content_list.setWordWrap(True)
        self.content_list.setTextElideMode(Qt.TextElideMode.ElideNone)
        self.content_list.itemClicked.connect(self._select_pdf)

        layout.addWidget(CaptionLabel("Zawartość folderu"))
        layout.addLayout(header_layout)
        layout.addWidget(self.content_list)
        return view

    def _create_pdf_view(self) -> QWidget:
        view = QWidget()
        layout = QVBoxLayout(view)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.addWidget(CaptionLabel("Wybrany plik PDF"))
        self.selected_pdf_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self.selected_pdf_label)
        self.image_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.image_list.setIconSize(QSize(192, 192))
        self.image_list.setGridSize(QSize(240, 240))
        self.image_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.image_list.setMovement(QListWidget.Movement.Static)
        self.image_list.setSpacing(12)
        self.image_list.setWordWrap(True)
        # ExtendedSelection enables Ctrl/Shift multi-select for batch export.
        self.image_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        layout.addWidget(self.image_list, 1)

        enhancement_options = QVBoxLayout()
        enhancement_options.addWidget(CaptionLabel("Poprawa zaznaczonych obrazów"))
        enhancement_row = QHBoxLayout()
        enhancement_row.addWidget(CaptionLabel("Metoda"))
        self.enhancement_combo.addItems([label for label, _ in ENHANCEMENT_METHODS])
        self.enhancement_combo.currentIndexChanged.connect(self._update_enhancement_controls)
        enhancement_row.addWidget(self.enhancement_combo)
        enhancement_row.addWidget(CaptionLabel("Tryb rozmiaru"))
        self.enhancement_resize_mode_combo.addItems(["Współczynnik", "Szerokość docelowa"])
        self.enhancement_resize_mode_combo.currentIndexChanged.connect(
            self._update_enhancement_controls
        )
        enhancement_row.addWidget(self.enhancement_resize_mode_combo)
        enhancement_row.addWidget(CaptionLabel("Wartość"))
        self.enhancement_scale_combo.addItems(["1.5x", "2x", "4x"])
        self.enhancement_width_combo.addItems(["1000 px", "1500 px", "2000 px", "3000 px"])
        enhancement_row.addWidget(self.enhancement_scale_combo)
        enhancement_row.addWidget(self.enhancement_width_combo)
        enhancement_row.addStretch()
        enhancement_options.addLayout(enhancement_row)

        model_row = QHBoxLayout()
        model_row.addWidget(CaptionLabel("Model"))
        self.enhancement_model_combo.addItems([name for name, _ in SUPER_IMAGE_MODELS])
        model_row.addWidget(self.enhancement_model_combo)
        model_row.addWidget(CaptionLabel("Urządzenie"))
        self.enhancement_device_combo.addItems(list(DEVICE_MAP))
        model_row.addWidget(self.enhancement_device_combo)
        model_row.addWidget(CaptionLabel("Maska"))
        self.enhancement_unsharp_combo.addItems(["1 / 100 / 0", "2 / 150 / 3", "3 / 200 / 5"])
        model_row.addWidget(self.enhancement_unsharp_combo)
        model_row.addStretch()
        enhancement_options.addLayout(model_row)
        enhancement_options.addWidget(
            CaptionLabel(
                "Opcja zostanie zastosowana wsadowo tuż przed zapisem plików. "
                "Maska: promień / siła / próg."
            )
        )
        layout.addLayout(enhancement_options)

        export_options = QHBoxLayout()
        export_options.addWidget(CaptionLabel("Format wyjściowy"))
        self.output_format_combo.addItems(list(OUTPUT_FORMAT_MAP))
        export_options.addWidget(self.output_format_combo)
        export_options.addStretch()
        layout.addLayout(export_options)

        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._export_selected_images)
        layout.addWidget(self.export_button)
        self._update_enhancement_controls()
        return view

    def _update_enhancement_controls(self) -> None:
        """Enable only controls supported by the selected enhancement backend."""
        method = ENHANCEMENT_METHODS[self.enhancement_combo.currentIndex()][1]
        uses_scaling = method in {EnhancementMethod.LANCZOS, EnhancementMethod.SUPER_IMAGE}
        super_image_selected = method is EnhancementMethod.SUPER_IMAGE

        # Pretrained super-image models only support integer 2x/4x scales, not 1.5x.
        self.enhancement_scale_combo.setItemEnabled(0, not super_image_selected)
        if super_image_selected and self.enhancement_scale_combo.currentIndex() == 0:
            self.enhancement_scale_combo.setCurrentIndex(1)

        uses_width_mode = uses_scaling and self.enhancement_resize_mode_combo.currentIndex() == 1
        self.enhancement_resize_mode_combo.setEnabled(uses_scaling)
        self.enhancement_scale_combo.setEnabled(uses_scaling and not uses_width_mode)
        self.enhancement_width_combo.setEnabled(uses_width_mode)
        self.enhancement_model_combo.setEnabled(super_image_selected)
        self.enhancement_device_combo.setEnabled(super_image_selected)
        self.enhancement_unsharp_combo.setEnabled(method is EnhancementMethod.UNSHARP_MASK)

    def _current_enhancement_options(self) -> EnhancementOptions:
        """Build an EnhancementOptions snapshot from the active UI controls."""
        method = ENHANCEMENT_METHODS[self.enhancement_combo.currentIndex()][1]
        resize_mode = (
            ResizeMode.WIDTH
            if self.enhancement_resize_mode_combo.currentIndex() == 1
            else ResizeMode.FACTOR
        )
        radius, percent, threshold = _parse_unsharp_preset(
            self.enhancement_unsharp_combo.currentText()
        )
        return EnhancementOptions(
            method=method,
            resize_mode=resize_mode,
            scale=float(self.enhancement_scale_combo.currentText().removesuffix("x")),
            target_width=int(self.enhancement_width_combo.currentText().removesuffix(" px")),
            model_name=self.enhancement_model_combo.currentText(),
            device=DEVICE_MAP[self.enhancement_device_combo.currentText()],
            unsharp_radius=radius,
            unsharp_percent=percent,
            unsharp_threshold=threshold,
        )

    def open_folder(self) -> None:
        """Switch back to the embedded folder picker view."""
        self.switchTo(self._folder_picker)

    def _select_pdf(self, item: QListWidgetItem) -> None:
        self.selected_pdf = Path(item.data(Qt.ItemDataRole.UserRole))
        self.selected_pdf_label.setText(self.selected_pdf.name)
        self._show_pdf_images()
        self._pdf_navigation.setEnabled(True)
        self.switchTo(self._pdf_view)

    def _show_pdf_images(self) -> None:
        """Extract the selected PDF's images and display their thumbnails."""
        if self.selected_pdf is None:
            return

        if self._extraction_job is not None:
            self._extraction_job.cancel()
            self._extraction_job = None

        self.image_list.clear()
        self.extracted_images = []
        self.export_button.setEnabled(False)
        job = PDFExtractionJob(self.selected_pdf, parent=self)
        self._extraction_job = job
        job.completed.connect(lambda images: self._show_extracted_images(job, images))
        job.failed.connect(lambda message: self._show_extraction_error(job, message))
        job.start()

    def _show_extracted_images(self, job: PDFExtractionJob, images: list[ExtractedImage]) -> None:
        """Display images only when they belong to the currently selected PDF."""
        # Guard against stale signals arriving after the user switched to another PDF.
        if self._extraction_job is not job:
            return
        self._extraction_job = None
        job.deleteLater()
        self.extracted_images = images
        for image_position, image in enumerate(images):
            pixmap = QPixmap()
            if not pixmap.loadFromData(image.thumbnail, "PNG"):
                MessageBox(
                    "Nie można wyświetlić miniatury",
                    f"Miniatura obrazu {image.image_index} na stronie "
                    f"{image.page_number} jest nieprawidłowa.",
                    self,
                ).exec()
                continue

            icon = QIcon(_fit_thumbnail(pixmap, QSize(192, 192)))
            item = QListWidgetItem(
                icon,
                f"Strona {image.page_number}, obraz {image.image_index}\n"
                f"{image.width} x {image.height} px",
            )
            item.setSizeHint(self.image_list.gridSize())
            item.setTextAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
            item.setData(Qt.ItemDataRole.UserRole, image_position)
            self.image_list.addItem(item)

        if not images:
            self.image_list.addItem(QListWidgetItem("Brak osadzonych obrazów"))
        else:
            self.export_button.setEnabled(True)

    def _show_extraction_error(self, job: PDFExtractionJob, message: str) -> None:
        """Display extraction errors for the currently selected PDF."""
        if self._extraction_job is not job:
            return
        self._extraction_job = None
        job.deleteLater()
        MessageBox("Nie można odczytać pliku PDF", message, self).exec()

    def _export_selected_images(self) -> None:
        """Save the selected images using the chosen enhancement and output format."""
        selected_items = self.image_list.selectedItems()
        if not selected_items:
            MessageBox("Brak zaznaczenia", "Zaznacz co najmniej jeden obraz.", self).exec()
            return

        output_path = QFileDialog.getExistingDirectory(self, "Wybierz folder eksportu")
        if not output_path:
            return

        destination = Path(output_path)
        output_format = OUTPUT_FORMAT_MAP.get(self.output_format_combo.currentText(), "original")
        enhancement_options = self._current_enhancement_options()
        selected_images = [
            self.extracted_images[item.data(Qt.ItemDataRole.UserRole)] for item in selected_items
        ]

        try:
            workers = min(len(selected_images), os.cpu_count() or 1)
            enhanced_batch = enhance_images(
                [image.data for image in selected_images],
                enhancement_options,
                workers=workers,
            )
            for image, enhanced_data in zip(selected_images, enhanced_batch, strict=True):
                if enhancement_options.method is EnhancementMethod.NONE:
                    image_data, extension = encode_image(image, output_format)
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
                    image_data, extension = encode_image(enhanced_image, output_format)

                filename = (
                    f"page-{image.page_number:04d}-image-{image.image_index:04d}.{extension}"
                )
                (destination / filename).write_bytes(image_data)
        except (EnhancementError, ImageConversionError, OSError) as error:
            MessageBox("Eksport nie powiódł się", str(error), self).exec()
            return

        MessageBox(
            "Eksport zakończony",
            f"Zapisano {len(selected_images)} obrazów w folderze {output_path}.",
            self,
        ).exec()

    def _show_folder_content(self, folder_path: Path) -> None:
        try:
            entries = sorted(
                folder_path.iterdir(), key=lambda path: (not path.is_dir(), path.name.lower())
            )
        except OSError as error:
            MessageBox("Nie można odczytać folderu", str(error), self).exec()
            return

        self.folder_label.setText(str(folder_path))
        self.content_list.clear()
        self.selected_pdf = None
        self.selected_pdf_label.setText("Nie wybrano pliku PDF")
        self._pdf_navigation.setEnabled(False)
        icon_provider = QFileIconProvider()
        font_metrics = self.content_list.fontMetrics()
        for entry in entries:
            if entry.suffix.lower() != ".pdf":
                continue
            display_name = _wrap_pdf_filename(entry.name, font_metrics)
            item = QListWidgetItem(icon_for_path(entry, icon_provider), display_name)
            item.setSizeHint(
                _pdf_tile_size(display_name, font_metrics, self.content_list.iconSize())
            )
            item.setTextAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
            item.setData(Qt.ItemDataRole.UserRole, str(entry))
            item.setToolTip(str(entry))
            self.content_list.addItem(item)

        self.switchTo(self._content_view)


def main() -> int:
    """Launch the Qt desktop application."""
    app = QApplication(sys.argv)
    apply_default_config()
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
