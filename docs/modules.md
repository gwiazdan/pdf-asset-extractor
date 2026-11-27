# Modules and Codebase Structure

This document covers every module under `src/pdf_asset_extractor/` and `tests/`.

## Package `pdf_asset_extractor`

### `config.py`

Defines application-wide constants and default QFluentWidgets styling.

- **Constants**:
  - `DEFAULT_FONT_FAMILIES` — fallback UI font stack (`Segoe UI`, `Microsoft YaHei`,
    `PingFang SC`).
  - `DEFAULT_THEME` — `Theme.AUTO` (follows OS light/dark theme).
  - `ACCENT_COLOR` — OS accent color on Windows/macOS, or `#0078D4` on Linux.
  - `ACRYLIC_ENABLED` — enables the window acrylic effect on Windows only
    (`sys.platform == "win32"`).
  - `BASE_DIR`, `LOGO_PATH`, `PDF_ICON_PATH` — resolved paths to assets in `assets/`.
  - `PDF_TILE_SIZE` — default PDF icon tile size (`180 x 180 px`).
- **Functions**:
  - `apply_default_config()` — configures font families, theme, and accent color in
    `qfluentwidgets`.

### `extractor.py`

Handles embedded PDF image extraction, thumbnail generation, and export format encoding.

- **Data models and exceptions**:
  - `ExtractedImage` (`@dataclass(frozen=True, slots=True)`) — immutable record containing
    `page_number`, `image_index`, `width`, `height`, `extension`, `data` (raw bytes), `thumbnail`
    (PNG bytes), and optional `path` / `thumbnail_path`.
  - `PDFExtractionError`, `ImageConversionError` — domain exceptions.
- **Public functions**:
  - `extract_images_with_thumbnails(pdf_path, output_dir=None, *, password=None,
    thumbnail_size=128) -> list[ExtractedImage]` — opens the PDF via PyMuPDF, optionally
    authenticates with `password`, iterates through pages and placed images (`page.get_images()`),
    extracts raw streams (`document.extract_image(...)`), builds PNG thumbnails, and optionally
    writes both files to `output_dir`.
  - `encode_image(image, output_format) -> tuple[bytes, str]` — returns raw bytes as-is for
    `"original"` or re-encodes via `fitz.Pixmap` to `"png"` or `"jpg"` (stripping alpha and
    converting non-RGB colorspaces to RGB for JPEG).
- **Internal helpers**:
  - `_ensure_png_compatible(pixmap, *, drop_alpha=False)` — converts non-grayscale/non-RGB
    colorspaces (such as CMYK) to RGB before PNG encoding.
  - `_thumbnail(image_data, max_size)` — builds a PNG thumbnail using power-of-two downsampling
    (`fitz.Pixmap.shrink`).
  - `_output_paths(output_dir, page_number, image_index, extension)` — produces distinct
    deterministic names `page-XXXX-image-YYYY.<ext>` (full image) and
    `page-XXXX-image-YYYY-thumb.png` (thumbnail).

### `extraction_worker.py`

Standalone subprocess CLI entrypoint invoked as
`python -m pdf_asset_extractor.extraction_worker <pdf_path> <output_dir>`.

- Runs `extract_images_with_thumbnails` with `thumbnail_size=256`.
- Writes a JSON manifest describing each extracted image and its filenames in `output_dir` to
  `stdout`.
- Catches `PDFExtractionError`, prints the message to `stderr`, and exits with code `1`.

### `image_processing.py`

Low-level module containing from-scratch NumPy implementations of resampling and sharpening (no
`PIL.Image.resize` or `PIL.ImageFilter`).

- **Public functions**:
  - `lanczos_resize(image, size, *, lobes=3) -> Image.Image` — separable Lanczos resampling
    (default Lanczos-3) supporting upscaling, downscaling, `L`/`LA`/`RGB`/`RGBA` preservation,
    and premultiplied alpha.
  - `unsharp_mask(image, *, radius=2.0, percent=150, threshold=3) -> Image.Image` — manual
    Gaussian unsharp mask with separable 1D convolution, detail thresholding, and untouched alpha
    channel preservation.
- **Internal helpers**:
  - `_normalize_image_mode`, `_to_float_channels`, `_from_float_channels` — safe conversion
    between `PIL.Image.Image` (including `P`, `1`, and `CMYK` modes) and contiguous `float32`
    `(H, W, C)` arrays.
  - `_sinc`, `_lanczos_kernel`, `_lanczos_weights`, `_resample_axis` — `float64` mathematical
    primitives and reusable-buffer 1D passes for Lanczos resampling.
  - `_gaussian_kernel_1d`, `_gaussian_blur` — `float64` 1D Gaussian kernel construction and
    reusable-buffer separable 2D convolution.

### `image_enhancer.py`

High-level orchestration layer for enhancing extracted images before export.

- **Types and configuration**:
  - `EnhancementMethod(StrEnum)` — `NONE`, `LANCZOS`, `UNSHARP_MASK`, `SUPER_IMAGE`.
  - `ResizeMode(StrEnum)` — `FACTOR` (scale multiplier) or `WIDTH` (target width preserving
    aspect ratio).
  - `SUPER_IMAGE_MODELS` — supported pretrained models: `drln-bam` (`DrlnModel`), `edsr`
    (`EdsrModel`), `msrn` (`MsrnModel`).
  - `EnhancementOptions` (`@dataclass(frozen=True, slots=True)`) — shared batch configuration
    (`method`, `resize_mode`, `scale`, `target_width`, `model_name`, `device`, `unsharp_radius`,
    `unsharp_percent`, `unsharp_threshold`).
  - `EnhancementError`, `EnhancementOutOfMemoryError` — enhancement-layer exceptions.
- **Public functions**:
  - `enhance_image(image_data: bytes, options: EnhancementOptions) -> bytes` — applies the
    selected enhancement method and returns PNG-encoded bytes (or original bytes for `NONE`).
  - `enhance_images(image_data_batch, options, *, workers=1) -> list[bytes]` — processes a batch
    of images; routes `SUPER_IMAGE` to batched PyTorch inference and CPU-only methods to optional
    `ThreadPoolExecutor` parallelism.
- **Internal helpers**:
  - `_resize_dimensions` — computes aspect-preserving `(width, height)`.
  - `_enhance_with_super_image`, `_enhance_with_super_image_batch` — dimension-grouped tensor
    batching (preserving input order), PyTorch inference, rounded uint8 conversion, and optional
    width fitting via `lanczos_resize`.
  - `_resolve_device`, `_load_super_image_model` — device selection and model caching
    (`@lru_cache(maxsize=1)`).

## Package `pdf_asset_extractor.ui`

### `ui/main_window.py`

Main desktop window (`MainWindow(MSFluentWindow)`) and application entry function `main()`
(exposed as the `gui` console script in `pyproject.toml`).

- Coordinates three stacked views:
  1. `FolderPickerWidget` — initial directory selection.
  2. Folder content view — tile grid of `.pdf` files with filename wrapping (`_wrap_pdf_filename`)
     and dynamic tile sizing (`_pdf_tile_size`).
  3. PDF view — thumbnail grid (`_fit_thumbnail`), enhancement controls, output format selector,
     and export action.
- Dynamically enables/disables option controls (`_update_enhancement_controls`) based on the
  active method (e.g., disables `1.5x` when `Super-image` is selected; enables the mask preset
  combo only for unsharp mask).
- Runs batched enhancement (`enhance_images`), format conversion, and disk export in
  `_export_selected_images`, building options through `_current_enhancement_options` and
  `_parse_unsharp_preset`.

### `ui/folder_picker.py`

Embedded directory explorer widget (`FolderPickerWidget(QWidget)`).

- Lists subdirectories alongside non-interactive `.pdf` entries so users can see which folders
  contain PDFs before selecting them.
- Supports parent navigation (`_go_up`), entering subdirectories on double-click (`_open_item`),
  highlighting a subdirectory on single-click (`_select_item`), and emitting
  `folder_selected(Path)`.

### `ui/pdf_extraction.py`

Asynchronous extraction controller (`PDFExtractionJob(QObject)`).

- Manages the lifecycle of `QProcess`, its `TemporaryDirectory`, and the timeout `QTimer`
  (`PDF_EXTRACTION_TIMEOUT_SECONDS = 60`).
- Guarantees single emission of either `completed(list[ExtractedImage])` or `failed(str)` (via
  `_finished`) and always cleans up temporary files in `_finish`.

### `ui/file_utils.py`

- `icon_for_path(path, icon_provider)` — returns the custom `assets/pdf.svg` icon for `.pdf`
  files or falls back to the native OS icon via `QFileIconProvider`.

## Test Suite (`tests/`)

- `tests/conftest.py` — registers benchmark CLI options (`--run-benchmark`,
  `--benchmark-images-dir`, `--benchmark-csv`, `--benchmark-limit`, `--benchmark-scales`,
  `--benchmark-models`, `--benchmark-device`).
- `tests/benchmark.py` — opt-in restoration benchmark comparing `lanczos` against `super-image`
  models (`drln-bam`, `edsr`, `msrn`) at 2x and 4x with per-image MSE, MAE, and PSNR written to
  CSV, plus unit tests for option parsing and unreadable image handling.
- `tests/test_extractor.py` — unit tests for multi-page extraction, password-protected PDFs,
  corrupt PDF errors, `encode_image` (including RGBA/CMYK handling), and `extraction_worker` CLI.
- `tests/test_image_processing.py` — unit tests for manual Lanczos-3 and unsharp mask (kernel
  properties, closeness to Pillow reference output, premultiplied alpha, threshold behavior,
  argument validation, and image mode handling).
- `tests/test_image_enhancer.py` — integration tests for `enhance_image` / `enhance_images`
  (including assertions that `PIL.Image.resize` and `PIL.Image.filter` are never invoked,
  multithreaded CPU batching, mixed-dimension `super-image` batching, and CPU fallback).
- `tests/test_pdf_extraction_job.py` — end-to-end tests for `PDFExtractionJob` extraction output
  and forced timeout process termination.
- `tests/test_ui_helpers.py` — unit tests for `icon_for_path`, `_parse_unsharp_preset`,
  `_wrap_pdf_filename`, `_fit_thumbnail`, and `FolderPickerWidget` navigation/selection.
