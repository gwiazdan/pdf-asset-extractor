"""Filesystem icon lookup helpers for GUI views."""

from pathlib import Path

from PySide6.QtCore import QFileInfo
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QFileIconProvider

from pdf_asset_extractor.config import PDF_ICON_PATH


def icon_for_path(path: Path, icon_provider: QFileIconProvider) -> QIcon:
    """Return the custom PDF icon or fall back to the operating system icon."""
    if path.suffix.lower() == ".pdf" and PDF_ICON_PATH.is_file():
        return QIcon(str(PDF_ICON_PATH))
    return icon_provider.icon(QFileInfo(str(path)))
