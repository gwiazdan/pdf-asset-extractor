from pathlib import Path
import sys

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication,
    QFileIconProvider,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    CaptionLabel,
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
from pdf_asset_extractor.ui.file_utils import icon_for_path
from pdf_asset_extractor.ui.folder_picker import FolderPickerWidget


class MainWindow(MSFluentWindow):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()
        self.init_window()

    def init_window(self) -> None:
        self.resize(900, 600)
        self.setWindowTitle("PDF Asset Extractor")
        self.setWindowIcon(QIcon(str(LOGO_PATH)))
        self.titleBar.setFixedHeight(45)
        self.selected_pdf: Path | None = None

        self._folder_picker = FolderPickerWidget()
        self._folder_picker.setObjectName("folderInterface")
        self._folder_picker.folder_selected.connect(self._show_folder_content)
        self._content_view = self._create_content_view()
        self._content_view.setObjectName("folderContentInterface")
        self._pdf_view = self._create_pdf_view()
        self._pdf_view.setObjectName("pdfInterface")

        self.addSubInterface(self._folder_picker, FluentIcon.FOLDER, "Folder")
        self.stackedWidget.addWidget(self._content_view)
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
        layout.addStretch()
        return view

    def open_folder(self) -> None:
        self.switchTo(self._folder_picker)

    def _select_pdf(self, item: QListWidgetItem) -> None:
        self.selected_pdf = Path(item.data(Qt.ItemDataRole.UserRole))
        self.selected_pdf_label.setText(self.selected_pdf.name)
        self._pdf_navigation.setEnabled(True)
        self.switchTo(self._pdf_view)

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
