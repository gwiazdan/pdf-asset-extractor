"""Application defaults built on top of QFluentWidgets."""

from pathlib import Path
import sys
from typing import Final

from PySide6.QtCore import QSize
from PySide6.QtGui import QColor
from qfluentwidgets import Theme, setFontFamilies, setTheme, setThemeColor
from qframelesswindow.utils import getSystemAccentColor

DEFAULT_FONT_FAMILIES: Final[list[str]] = [
    "Segoe UI",
    "Microsoft YaHei",
    "PingFang SC",
]
DEFAULT_THEME: Final[Theme] = Theme.AUTO
# Native accent queries are only supported by qframelesswindow on Windows and macOS.
ACCENT_COLOR: Final[QColor] = (
    getSystemAccentColor() if sys.platform in {"win32", "darwin"} else QColor("#0078D4")
)
# Acrylic window backdrop is a Windows DWM feature.
ACRYLIC_ENABLED: Final[bool] = sys.platform == "win32"


def _resolve_base_dir() -> Path:
    """Return the bundle root when frozen by PyInstaller or the repository root."""
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass is not None:
        return Path(meipass)
    return Path(__file__).resolve().parent.parent.parent


BASE_DIR = _resolve_base_dir()
LOGO_PATH = BASE_DIR / "assets" / "logo.svg"
PDF_ICON_PATH = BASE_DIR / "assets" / "pdf.svg"
PDF_TILE_SIZE = QSize(180, 180)


def apply_default_config() -> None:
    """Apply the default application appearance."""
    setFontFamilies(DEFAULT_FONT_FAMILIES)
    setTheme(DEFAULT_THEME, lazy=True)
    setThemeColor(ACCENT_COLOR, lazy=True)
