from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFileIconProvider, QHBoxLayout, QListWidgetItem, QVBoxLayout, QWidget
from qfluentwidgets import (
    CaptionLabel,
    FluentIcon,
    ListWidget,
    PrimaryPushButton,
    PushButton,
    StrongBodyLabel,
)

from pdf_asset_extractor.ui.file_utils import icon_for_path


class FolderPickerWidget(QWidget):
    """Embedded folder picker with an explorer-like directory list."""

    folder_selected = Signal(Path)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.current_folder = Path.home()
        self.selected_folder: Path | None = None
        self.icon_provider = QFileIconProvider()
        self._build_ui()
        self._refresh()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)

        self.path_label = StrongBodyLabel()
        self.folder_list = ListWidget()
        self.folder_list.setViewMode(ListWidget.ViewMode.ListMode)
        self.folder_list.itemClicked.connect(self._select_item)
        self.folder_list.itemDoubleClicked.connect(self._open_item)

        navigation = QHBoxLayout()
        up_button = PushButton(FluentIcon.UP, "W górę")
        up_button.clicked.connect(self._go_up)
        navigation.addWidget(up_button)
        navigation.addWidget(self.path_label, 1)

        buttons = QHBoxLayout()
        select_button = PrimaryPushButton(FluentIcon.ACCEPT, "Wybierz folder")
        select_button.clicked.connect(self._select_folder)
        buttons.addStretch()
        buttons.addWidget(select_button)

        layout.addLayout(navigation)
        layout.addWidget(CaptionLabel("Foldery"))
        layout.addWidget(self.folder_list)
        layout.addLayout(buttons)

    def _refresh(self) -> None:
        self.path_label.setText(str(self.current_folder))
        self.folder_list.clear()
        self.selected_folder = None
        try:
            entries = sorted(
                (
                    path
                    for path in self.current_folder.iterdir()
                    if path.is_dir()
                ),
                key=lambda path: path.name.lower(),
            )
        except OSError:
            entries = []

        for entry in entries:
            item = QListWidgetItem(icon_for_path(entry, self.icon_provider), entry.name)
            item.setData(Qt.ItemDataRole.UserRole, str(entry))
            item.setToolTip(str(entry))
            self.folder_list.addItem(item)

    def _select_item(self, item: QListWidgetItem) -> None:
        entry = Path(item.data(Qt.ItemDataRole.UserRole))
        if entry.is_dir():
            self.selected_folder = entry

    def _open_item(self, item: QListWidgetItem) -> None:
        entry = Path(item.data(Qt.ItemDataRole.UserRole))
        if not entry.is_dir():
            return

        self.current_folder = entry
        self._refresh()

    def _go_up(self) -> None:
        parent = self.current_folder.parent
        if parent == self.current_folder:
            return
        self.current_folder = parent
        self._refresh()

    def _select_folder(self) -> None:
        self.folder_selected.emit(self.selected_folder or self.current_folder)
