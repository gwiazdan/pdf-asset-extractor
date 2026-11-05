import sys

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication
from qfluentwidgets import MSFluentWindow

from pdf_asset_extractor.config import ACRYLIC_ENABLED, LOGO_PATH, apply_default_config


class MainWindow(MSFluentWindow):
    """Main interface"""

    def __init__(self) -> None:
        super().__init__()
        self.init_window()

    def init_window(self) -> None:
        self.resize(900, 600)
        self.setWindowTitle("PDF Asset Extractor")
        self.setWindowIcon(QIcon(str(LOGO_PATH)))

        if ACRYLIC_ENABLED:
            try:
                self.windowEffect.setAcrylicEffect(self.winId())
            except Exception as e:
                print(f"Nie udało się wyłączyć efektu akrylowego: {e}")


def main() -> int:
    app = QApplication(sys.argv)
    apply_default_config()
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
