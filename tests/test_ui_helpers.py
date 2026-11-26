"""Unit tests for UI helper functions and FolderPickerWidget."""

import os
from pathlib import Path

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPixmap
from PySide6.QtWidgets import QApplication, QFileIconProvider
import pytest

from pdf_asset_extractor.ui.file_utils import icon_for_path
from pdf_asset_extractor.ui.folder_picker import FolderPickerWidget
from pdf_asset_extractor.ui.main_window import (
    _fit_thumbnail,
    _parse_unsharp_preset,
    _wrap_pdf_filename,
)


@pytest.fixture(scope="module", autouse=True)
def qt_gui_application() -> QApplication:
    """Ensure a QApplication exists for widget and pixmap tests, even headless."""
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    existing = QApplication.instance()
    if isinstance(existing, QApplication):
        return existing
    return QApplication([])


def test_parse_unsharp_preset_valid_and_invalid() -> None:
    assert _parse_unsharp_preset("2 / 150 / 3") == (2.0, 150, 3)
    assert _parse_unsharp_preset("1.5 / 100 / 0") == (1.5, 100, 0)

    with pytest.raises(ValueError, match="Invalid unsharp mask preset"):
        _parse_unsharp_preset("2 / 150")


def test_fit_thumbnail_centers_and_preserves_canvas_size() -> None:
    source = QPixmap(80, 40)
    source.fill(QColor("red"))

    fitted = _fit_thumbnail(source, QSize(100, 100))

    assert fitted.size() == QSize(100, 100)
    rendered = fitted.toImage()
    # Top/bottom margins should be transparent while center pixel is opaque red.
    assert rendered.pixelColor(50, 2).alpha() == 0
    assert rendered.pixelColor(50, 50) == QColor("red")


def test_wrap_pdf_filename_breaks_long_unbroken_names() -> None:
    metrics = QFontMetrics(QFont())
    long_name = "a" * 80 + ".pdf"

    wrapped = _wrap_pdf_filename(long_name, metrics)

    assert "\n" in wrapped
    assert wrapped.replace("\n", "") == long_name


def test_icon_for_path_returns_non_null_icon_for_pdf_and_folder(tmp_path: Path) -> None:
    pdf_file = tmp_path / "doc.pdf"
    pdf_file.write_bytes(b"%PDF-1.4")
    provider = QFileIconProvider()

    pdf_icon = icon_for_path(pdf_file, provider)
    dir_icon = icon_for_path(tmp_path, provider)

    assert not pdf_icon.isNull()
    assert dir_icon is not None


def test_folder_picker_lists_dirs_and_readonly_pdfs_and_emits_selection(tmp_path: Path) -> None:
    subdir = tmp_path / "alpha"
    subdir.mkdir()
    pdf_file = tmp_path / "sample.pdf"
    pdf_file.write_bytes(b"%PDF-1.4")
    ignored_txt = tmp_path / "notes.txt"
    ignored_txt.write_text("ignore me", encoding="utf-8")

    picker = FolderPickerWidget()
    picker.current_folder = tmp_path
    picker._refresh()

    item_names = [picker.folder_list.item(i).text() for i in range(picker.folder_list.count())]
    assert item_names == ["alpha", "sample.pdf"]

    # Directory items remain selectable, while PDF items are non-interactive hints.
    dir_item = picker.folder_list.item(0)
    pdf_item = picker.folder_list.item(1)
    assert dir_item.flags() != Qt.ItemFlag.NoItemFlags
    assert pdf_item.flags() == Qt.ItemFlag.NoItemFlags

    emitted: list[Path] = []
    picker.folder_selected.connect(emitted.append)

    # Selecting a subdirectory and confirming emits the subdirectory.
    picker._select_item(dir_item)
    picker._select_folder()
    assert emitted == [subdir]

    # Double-clicking enters the subdirectory, and _go_up returns to tmp_path.
    picker._open_item(dir_item)
    assert picker.current_folder == subdir
    picker._go_up()
    assert picker.current_folder == tmp_path
