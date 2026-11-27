# pdf-asset-extractor

Desktop utility (PySide6 + QFluentWidgets) for browsing folders of PDF files,
extracting embedded images with thumbnails, and batch-exporting them with
optional enhancement (manual Lanczos-3 resampling, manual unsharp mask, or
`super-image` neural upscaling).

## Documentation

Detailed technical documentation is available in `docs/`:

- [Overview & quick start](docs/README.md)
- [Architecture](docs/architecture.md)
- [Modules & codebase structure](docs/modules.md)
- [Libraries & dependencies](docs/libraries.md)
- [Workflow & algorithms](docs/workflow.md)

## Restoration benchmark

The image restoration benchmark is opt-in. By default, `pytest` skips it. To
compare Lanczos with DRLN-BAM, EDSR, and MSRN at 2x and 4x:

```shell
pytest -s --run-benchmark
```

By default, images are read from `images/` and per-image MSE, MAE, and PSNR
results are written to `output/benchmark.csv`. The benchmark first downscales
each source image, embeds it in a temporary PDF, and extracts it through the
application extractor. The `-s` flag lets pytest display the tqdm progress bars.
Unreadable images are reported in the CSV's `error` column and do not stop the
remaining benchmark.
PDF preparation and enhancement run sequentially, one image at a time.

Useful options include `--benchmark-limit 10` for a small trial run,
`--benchmark-scales 2` to run only one scale, and
`--benchmark-models drln-bam,edsr` to select a subset of models
(`--benchmark-models none` runs Lanczos only). Use
`--benchmark-device cpu` to force CPU inference. PyTorch CPU inference uses one
thread.

## Building standalone binaries

Standalone `--onedir` bundles (PyInstaller + **ONNX Runtime** with **DirectML GPU**
acceleration on Windows and CPU/CUDA on Linux, without bundling PyTorch) are built
automatically by `.github/workflows/build.yml` for **Windows (x64)** and **Linux (x64)**:

- Push to `main`, pull requests, and manual `workflow_dispatch` runs upload
  verified `.zip` / `.tar.gz` artifacts.
- Pushing a version tag (`v*`, e.g. `git tag v0.1.0 && git push origin v0.1.0`)
  publishes a GitHub Release with both archives and `SHA256SUMS.txt`.
- Run the **Export ONNX Models** workflow (`.github/workflows/export-models.yml`)
  once from the GitHub **Actions** tab to publish the `models-v1` Release containing
  the 6 `.onnx` super-resolution model files downloaded on demand by the app.

To build locally on Linux:

```shell
uv sync --no-dev --group build
uv run --no-sync python scripts/build.py
```

To build locally on Windows with DirectML GPU support:

```shell
uv sync --no-dev --group build
uv pip uninstall --python .venv onnxruntime
uv pip install --no-config --python .venv "onnxruntime-directml>=1.20.0"
uv run --no-sync python scripts/build.py --require-directml
```
