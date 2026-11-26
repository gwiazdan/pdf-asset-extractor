from pathlib import Path


def pytest_addoption(parser) -> None:
    """Register options for the opt-in restoration benchmark."""
    group = parser.getgroup("Benchmark")
    group.addoption(
        "--run-benchmark",
        action="store_true",
        default=False,
        help="run the image restoration benchmark",
    )
    group.addoption(
        "--benchmark-images-dir",
        type=Path,
        default=Path("images"),
        help="directory containing source images",
    )
    group.addoption(
        "--benchmark-csv",
        type=Path,
        default=Path("output/benchmark.csv"),
        help="CSV file for per-image benchmark results",
    )
    group.addoption(
        "--benchmark-limit",
        type=int,
        default=0,
        help="limit the benchmark to the first N images (0 means all)",
    )
    group.addoption(
        "--benchmark-scales",
        default="2,4",
        help="comma-separated downscale/upscale factors (2 and/or 4)",
    )
    group.addoption(
        "--benchmark-models",
        default="all",
        help="comma-separated super-image model identifiers or 'all'",
    )
    group.addoption(
        "--benchmark-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="device used by super-image models",
    )
