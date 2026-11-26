# Libraries and Dependencies

All project dependencies are declared in [pyproject.toml](../pyproject.toml) and managed with
**uv** (locked in `uv.lock`, targeting Python `>= 3.12`).

## Runtime Dependencies (`dependencies`)

| Library | Version | Used In | Role in the Project |
| --- | --- | --- | --- |
| **PySide6** | `>=6.10.0` | `config.py`, `ui/*`, `tests/test_pdf_extraction_job.py` | Qt 6 GUI framework: windows, layouts, widgets, image rendering (`QPixmap`, `QIcon`), event loop, signals/slots, and subprocess control (`QProcess`, `QTimer`). |
| **pyside6-fluent-widgets[full]** | `>=1.11.3` | `config.py`, `ui/main_window.py`, `ui/folder_picker.py` | Fluent Design UI components (`MSFluentWindow`, `ComboBox`, `ListWidget`, `PushButton`, `MessageBox`, `FluentIcon`, system theme/accent integration, and Windows acrylic backdrop). |
| **PyMuPDF (`pymupdf` / `fitz`)** | `>=2.1` | `extractor.py`, `tests/benchmark.py`, `tests/test_pdf_extraction_job.py` | PDF parsing, password authentication, raw embedded image extraction (`extract_image`), fast thumbnail shrinking (`Pixmap.shrink`), and PNG/JPEG encoding. |
| **NumPy** | `>=2.0` | `image_processing.py`, `image_enhancer.py`, `tests/*` | Vectorized `float64` numerical engine for the manual **Lanczos-3** and **Unsharp Mask** algorithms, PyTorch tensor-to-image conversion, and benchmark metric computation (MSE/MAE/PSNR). |
| **Pillow (`PIL`)** | `>=12.0` | `image_processing.py`, `image_enhancer.py`, `tests/*` | Image container decoding/encoding (`Image.open`, `Image.fromarray`, `Image.save`) and colorspace normalization (`L`, `LA`, `RGB`, `RGBA`). **Not** used for Lanczos resampling or unsharp masking in production code (used in tests only as a reference baseline and fixture generator). |
| **super-image** | `>=0.2.0` | `image_enhancer.py` (lazy import) | Pretrained Hugging Face super-resolution models (`eugenesiow/drln-bam`, `eugenesiow/edsr`, `eugenesiow/msrn`) and `ImageLoader` for image-to-tensor preprocessing. Brings in **PyTorch (`torch`)** transitively. |
| **opencv-python** | `<5` | `super-image` runtime ecosystem | Dependency required by the `super-image` stack (not imported directly in `src/pdf_asset_extractor/`). |
| **tqdm** | `>=4.70.1` | `tests/benchmark.py` | Terminal progress bars during benchmark PDF preparation and evaluation. |
| **pytest** | `>=9.2.0` | `tests/*` | Test runner for unit tests, integration tests, and the opt-in restoration benchmark. |

### Transitive Dependencies Used Directly

- **PyTorch (`torch`)** — installed via `super-image` and imported lazily inside
  `image_enhancer.py` and `tests/benchmark.py` for CUDA detection (`torch.cuda.is_available`),
  batch concatenation (`torch.cat`), gradient-free inference (`torch.inference_mode()`),
  `torch.cuda.OutOfMemoryError` handling, and single-threaded CPU execution during benchmarking
  (`torch.set_num_threads(1)`). Lazy imports keep GUI startup and CPU-only methods lightweight.

## Python Standard Library

Key standard-library modules used across the codebase include:

- `dataclasses`, `enum.StrEnum`, `pathlib.Path` — typed immutable records and filesystem paths;
- `concurrent.futures.ThreadPoolExecutor`, `functools.lru_cache`, `functools.partial` — CPU batch
  parallelism and single-slot model caching;
- `contextlib.ExitStack`, `io.BytesIO`, `tempfile.TemporaryDirectory` — deterministic resource
  management in memory and on disk;
- `argparse`, `json`, `csv`, `math`, `sys` — worker CLI, IPC manifest serialization, benchmark
  reporting, and math utilities.

## Development & Quality Tools (`dependency-groups`)

| Group | Tool | Purpose |
| --- | --- | --- |
| `dev` | **pre-commit** (`>=4.7.0`) | Runs git hooks before commits (conventional commits, whitespace/EOF checks, Ruff, rumdl). |
| `type_check` | **basedpyright** (`>=1.40.0`) | Static type checking for Python code. |
| `lint` | **ruff** | Python linter (`F`, `E`, `W`, `I`, `N`, `UP`, `RUF`, `B`, `C4`, `ISC`, `PIE`, `PT`, `PTH`, `SIM`, `TID`) and formatter with a 99-character line limit. |
| `lint` | **rumdl** | Markdown linter and formatter with a 99-character line limit. |

## Build Backend

The package is built with **hatchling** (`>=1.27.0`) and exposes the desktop application script:

```toml
[project.scripts]
gui = "pdf_asset_extractor.ui.main_window:main"
```
