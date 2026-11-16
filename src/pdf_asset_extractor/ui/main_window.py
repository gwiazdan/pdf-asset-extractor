from pathlib import Path
import sys

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
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
from pdf_asset_extractor.extractor import (
    ExtractedImage,
    ImageConversionError,
    PDFExtractionError,
    encode_image,
    extract_images_with_thumbnails,
)
from pdf_asset_extractor.ui.file_utils import icon_for_path
from pdf_asset_extractor.ui.folder_picker import FolderPickerWidget


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


class MainWindow(MSFluentWindow):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()
        self.init_window()

    def init_window(self) -> None:
        self.resize(1024, 768)
        self.setWindowTitle("PDF Asset Extractor")
        self.setWindowIcon(QIcon(str(LOGO_PATH)))
        self.titleBar.setFixedHeight(45)
        self.selected_pdf: Path | None = None
        self.extracted_images: list[ExtractedImage] = []

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
                print(f"Nie udało się wyłączyć efektu akrylowego: {error}")

    def _create_content_view(self) -> QWidget:
        view = QWidget()
        layout = QVBoxLayout(view)
        layout.setContentsMargins(24, 24, 24, 24)

        header_layout = QHBoxLayout()
        self.folder_label = StrongBodyLabel()
        self.folder_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        choose_button = PushButton(FluentIcon.FOLDER_ADD, "Wybierz inny folder")
        choose_button.clicked.connect(self.open_folder)
        header_layout.addWidget(self.folder_label, 1)
        header_layout.addWidget(choose_button)

        self.content_list = ListWidget()
        self.content_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.content_list.setIconSize(QSize(64, 64))
        self.content_list.setGridSize(PDF_TILE_SIZE)
        self.content_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.content_list.setMovement(QListWidget.Movement.Static)
        self.content_list.setSpacing(12)
        self.content_list.setWordWrap(True)
        self.content_list.setTextElideMode(Qt.TextElideMode.ElideNone)
        self.content_list.setUniformItemSizes(True)
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
        self.selected_pdf_label = StrongBodyLabel("Nie wybrano pliku PDF")
        self.selected_pdf_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self.selected_pdf_label)
        self.image_list = ListWidget()
        self.image_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.image_list.setIconSize(QSize(192, 192))
        self.image_list.setGridSize(QSize(240, 240))
        self.image_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.image_list.setMovement(QListWidget.Movement.Static)
        self.image_list.setSpacing(12)
        self.image_list.setWordWrap(True)
        self.image_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        layout.addWidget(self.image_list, 1)

        export_options = QHBoxLayout()
        export_options.addWidget(CaptionLabel("Format wyjściowy"))
        self.output_format_combo = ComboBox()
        self.output_format_combo.addItems(["Oryginalny", "PNG", "JPEG"])
        export_options.addWidget(self.output_format_combo)
        export_options.addStretch()
        layout.addLayout(export_options)

        self.export_button = PushButton("Eksportuj zaznaczone obrazy")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._export_selected_images)
        layout.addWidget(self.export_button)
        return view

    def open_folder(self) -> None:
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

        self.image_list.clear()
        self.extracted_images = []
        self.export_button.setEnabled(False)
        try:
            images = extract_images_with_thumbnails(self.selected_pdf, thumbnail_size=256)
        except PDFExtractionError as error:
            MessageBox("Nie można odczytać pliku PDF", str(error), self).exec()
            return

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

    def _export_selected_images(self) -> None:
        """Save the selected images using the chosen output format."""
        selected_items = self.image_list.selectedItems()
        if not selected_items:
            MessageBox("Brak zaznaczenia", "Zaznacz co najmniej jeden obraz.", self).exec()
            return

        output_path = QFileDialog.getExistingDirectory(self, "Wybierz folder eksportu")
        if not output_path:
            return

        output_format = {"Oryginalny": "original", "PNG": "png", "JPEG": "jpg"}.get(
            self.output_format_combo.currentText(), "original"
        )
        try:
            for item in selected_items:
                image = self.extracted_images[item.data(Qt.ItemDataRole.UserRole)]
                image_data, extension = encode_image(image, output_format)
                filename = (
                    f"page-{image.page_number:04d}-image-"
                    f"{image.image_index:04d}.{extension}"
                )
                (Path(output_path) / filename).write_bytes(image_data)
        except (ImageConversionError, OSError) as error:
            MessageBox("Eksport nie powiódł się", str(error), self).exec()
            return

        MessageBox(
            "Eksport zakończony",
            f"Zapisano {len(selected_items)} obrazów w folderze {output_path}.",
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
        for entry in entries:
            if entry.suffix.lower() != ".pdf":
                continue
            item = QListWidgetItem(icon_for_path(entry, icon_provider), entry.name)
            item.setSizeHint(PDF_TILE_SIZE)
            item.setTextAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
            item.setData(Qt.ItemDataRole.UserRole, str(entry))
            item.setToolTip(str(entry))
            self.content_list.addItem(item)

        self.switchTo(self._content_view)


def main() -> int:
    app = QApplication(sys.argv)
    apply_default_config()
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
