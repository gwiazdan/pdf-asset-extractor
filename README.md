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
