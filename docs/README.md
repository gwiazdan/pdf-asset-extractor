# PDF Asset Extractor Documentation

**PDF Asset Extractor** is a Python desktop application (PySide6 + QFluentWidgets) for browsing
PDF files in a selected directory, extracting embedded images with thumbnails, and batch-exporting
them with optional enhancement (manual Lanczos-3 resampling, manual unsharp mask, or neural
super-resolution via `super-image`).

## Contents

- [System Architecture](architecture.md) — application layers, inter-process communication (IPC),
  Mermaid diagrams, concurrency model, and exception hierarchy.
- [Modules and Codebase Structure](modules.md) — breakdown of every module in
  `src/pdf_asset_extractor/` and `tests/`, including responsibilities, classes, and functions.
- [Libraries and Dependencies](libraries.md) — runtime, transitive, standard-library, and
  development dependencies with their exact roles in the project.
- [Workflow and Algorithms](workflow.md) — end-to-end user flow, mathematical details of the
  manual Lanczos-3 and unsharp mask implementations, `super-image` pipeline, format encoding, and
  restoration benchmark.

## Project Layout

```text
pdf-asset-extractor/
├── assets/                          # GUI vector assets (logo.svg, pdf.svg)
├── docs/                            # Technical documentation
├── images/                          # Source image dataset for the benchmark
├── output/                          # Benchmark output (default: output/benchmark.csv)
├── src/pdf_asset_extractor/
│   ├── config.py                    # Appearance defaults and path/tile constants
│   ├── extractor.py                 # PDF image extraction, thumbnails, and format encoding
│   ├── extraction_worker.py         # Subprocess CLI entrypoint for PDF extraction (JSON manifest)
│   ├── image_processing.py          # Manual Lanczos-3 and unsharp mask implementations in NumPy
│   ├── image_enhancer.py            # Enhancement orchestration and super-image batch inference
│   └── ui/
│       ├── main_window.py           # Main application window (MSFluentWindow) and export flow
│       ├── folder_picker.py         # Embedded directory explorer widget
│       ├── pdf_extraction.py        # Subprocess controller (QProcess + 60 s timeout)
│       └── file_utils.py            # Custom PDF and OS file icon helper
├── tests/
│   ├── conftest.py                  # Custom pytest CLI flags for the benchmark
│   ├── benchmark.py                 # Opt-in restoration benchmark (MSE, MAE, PSNR)
│   ├── test_extractor.py            # Unit tests for PDF extraction, encoding, and worker CLI
│   ├── test_image_processing.py     # Unit tests for manual Lanczos-3 and unsharp mask
│   ├── test_image_enhancer.py       # Integration tests for the enhancement pipeline
│   ├── test_pdf_extraction_job.py   # Subprocess extraction and timeout tests
│   └── test_ui_helpers.py           # Unit tests for UI helpers and FolderPickerWidget
├── pyproject.toml                   # Project metadata, dependencies, and tool configuration
└── README.md                        # Project overview and benchmark usage
```

## Quick Start

```shell
# Install dependencies
uv sync

# Launch the desktop GUI
uv run gui

# Run unit and integration tests
uv run pytest

# Run the opt-in restoration benchmark on a small subset
uv run pytest -s --run-benchmark --benchmark-limit 10
```
