# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller specification for PDF Asset Extractor (--onedir)."""

from importlib.util import find_spec
from pathlib import Path
import sys

from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules, copy_metadata

ROOT_DIR = Path(SPECPATH).resolve().parent
ENTRYPOINT = ROOT_DIR / "src" / "pdf_asset_extractor" / "__main__.py"
ICON_PATH = ROOT_DIR / "build" / "app.ico"

binaries = [
    *collect_dynamic_libs("torchvision"),
]

# Torchvision loads native C++ operator extensions (e.g. _C_stable.so / .pyd) dynamically
# via torch.ops.load_library() rather than Python import, so collect them explicitly.
_tv_spec = find_spec("torchvision")
if _tv_spec is not None and _tv_spec.submodule_search_locations:
    _tv_dir = Path(next(iter(_tv_spec.submodule_search_locations)))
    for _pattern in ("*.so", "*.so.*", "*.pyd", "*.dll"):
        for _lib in sorted(_tv_dir.glob(_pattern)):
            binaries.append((str(_lib), "torchvision"))

datas = [
    (str(ROOT_DIR / "assets" / "logo.svg"), "assets"),
    (str(ROOT_DIR / "assets" / "pdf.svg"), "assets"),
]

for distribution in (
    "filelock",
    "huggingface-hub",
    "packaging",
    "requests",
    "super-image",
    "torch",
    "torchvision",
    "tqdm",
):
    try:
        datas += copy_metadata(distribution)
    except Exception:
        pass

hiddenimports = [
    "pdf_asset_extractor.extraction_worker",
    "pdf_asset_extractor.ui.main_window",
    "PySide6.QtSvg",
    "PySide6.QtSvgWidgets",
    "PySide6.QtXml",
    *collect_submodules("super_image"),
]

excludes = [
    # Unused Qt modules (the GUI only uses QtCore, QtGui, QtWidgets, QtSvg, QtXml).
    "PySide6.Qt3DAnimation",
    "PySide6.Qt3DCore",
    "PySide6.Qt3DExtras",
    "PySide6.Qt3DInput",
    "PySide6.Qt3DLogic",
    "PySide6.Qt3DRender",
    "PySide6.QtBluetooth",
    "PySide6.QtCharts",
    "PySide6.QtDataVisualization",
    "PySide6.QtDesigner",
    "PySide6.QtGraphs",
    "PySide6.QtHelp",
    "PySide6.QtLocation",
    "PySide6.QtMultimedia",
    "PySide6.QtMultimediaWidgets",
    "PySide6.QtNfc",
    "PySide6.QtPdf",
    "PySide6.QtPdfWidgets",
    "PySide6.QtPositioning",
    "PySide6.QtQml",
    "PySide6.QtQuick",
    "PySide6.QtQuick3D",
    "PySide6.QtQuickWidgets",
    "PySide6.QtRemoteObjects",
    "PySide6.QtScxml",
    "PySide6.QtSensors",
    "PySide6.QtSerialPort",
    "PySide6.QtSql",
    "PySide6.QtStateMachine",
    "PySide6.QtTest",
    "PySide6.QtTextToSpeech",
    "PySide6.QtVirtualKeyboard",
    "PySide6.QtWebChannel",
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineQuick",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebSockets",
    # Unused PyMuPDF layout/LLM extras.
    "onnxruntime",
    "pymupdf4llm",
    "pymupdf_layout",
    # Development, linting, and testing tools.
    "_pytest",
    "basedpyright",
    "nodejs_wheel",
    "pre_commit",
    "pytest",
    "ruff",
    "rumdl",
    "tkinter",
]

a = Analysis(
    [str(ENTRYPOINT)],
    pathex=[str(ROOT_DIR / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="pdf-asset-extractor",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=sys.platform != "win32",
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ICON_PATH) if sys.platform == "win32" and ICON_PATH.is_file() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="pdf-asset-extractor",
)
