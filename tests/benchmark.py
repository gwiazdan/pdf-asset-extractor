"""Opt-in image restoration benchmark."""

import csv
from dataclasses import dataclass
from io import BytesIO
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image
import pymupdf as fitz
import pytest
from tqdm import tqdm

from pdf_asset_extractor.extractor import PDFExtractionError, extract_images_with_thumbnails
from pdf_asset_extractor.image_enhancer import (
    SUPER_IMAGE_MODELS,
    EnhancementError,
    EnhancementMethod,
    EnhancementOptions,
    EnhancementOutOfMemoryError,
    enhance_image,
)


@dataclass(frozen=True, slots=True)
class BenchmarkSample:
    """A low-resolution image extracted from its temporary PDF."""

    image_path: Path
    scale: int
    extracted_path: Path
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class BenchmarkFailure:
    """A source image or PDF that could not be prepared."""

    image_path: Path
    scale: int
    error: str


@dataclass(frozen=True, slots=True)
class SampleJob:
    """One image/scale task in PDF preparation."""

    image_number: int
    image_path: Path
    scale: int
    workspace: Path


def _prepare_one_sample(job: SampleJob) -> BenchmarkSample | BenchmarkFailure:
    """Downscale one image, round-trip it through PDF extraction, and cache it."""
    try:
        with Image.open(job.image_path) as source:
            original = source.convert("RGB")
    except (OSError, ValueError) as error:
        return BenchmarkFailure(job.image_path, job.scale, f"{type(error).__name__}: {error}")

    with original:
        low_width = max(round(original.width / job.scale), 1)
        low_height = max(round(original.height / job.scale), 1)
        try:
            low_resolution = original.resize(
                (low_width, low_height),
                Image.Resampling.LANCZOS,
            )
        except (OSError, ValueError) as error:
            return BenchmarkFailure(
                job.image_path,
                job.scale,
                f"{type(error).__name__}: {error}",
            )

        encoded_image = BytesIO()
        low_resolution.save(encoded_image, format="PNG")

    pdf_path = job.workspace / "pdfs" / f"{job.image_number:04d}-scale-{job.scale}.pdf"
    extracted_path = (
        job.workspace
        / "extracted"
        / f"{job.image_number:04d}-{job.image_path.stem}-scale-{job.scale}.img"
    )
    try:
        try:
            with fitz.open() as document:
                page = document.new_page(width=low_width, height=low_height)
                page.insert_image(page.rect, stream=encoded_image.getvalue())
                document.save(pdf_path)

            extracted_images = extract_images_with_thumbnails(pdf_path)
            if len(extracted_images) != 1:
                raise PDFExtractionError(
                    f"Expected one embedded image in benchmark PDF for {job.image_path.name}"
                )
            extracted_path.write_bytes(extracted_images[0].data)
        except PDFExtractionError as error:
            return BenchmarkFailure(
                job.image_path,
                job.scale,
                f"{type(error).__name__}: {error}",
            )
    finally:
        pdf_path.unlink(missing_ok=True)

    return BenchmarkSample(
        image_path=job.image_path,
        scale=job.scale,
        extracted_path=extracted_path,
        width=original.width,
        height=original.height,
    )


def _parse_scales(value: str) -> list[int]:
    try:
        scales = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise pytest.UsageError("--benchmark-scales must contain 2 and/or 4") from error
    if not scales or any(scale not in {2, 4} for scale in scales):
        raise pytest.UsageError("--benchmark-scales must contain 2 and/or 4")
    return list(dict.fromkeys(scales))


def _parse_models(value: str) -> list[str]:
    available_models = [name for name, _ in SUPER_IMAGE_MODELS]
    normalized_value = value.strip().lower()
    if normalized_value == "all":
        return available_models
    if normalized_value == "none":
        return []

    models = list(dict.fromkeys(name.strip() for name in value.split(",") if name.strip()))
    unknown_models = sorted(set(models) - set(available_models))
    if not models or unknown_models:
        raise pytest.UsageError(
            "Use 'all', 'none', or valid model identifiers; "
            f"unknown identifiers: {unknown_models}"
        )
    return models


def test_parse_models_limits_available_models() -> None:
    assert _parse_models("all") == ["drln-bam", "edsr", "msrn"]

    with pytest.raises(pytest.UsageError, match="unknown identifiers"):
        _parse_models("edsr-base,pan")


def _prepare_samples(
    image_paths: list[Path], scales: list[int], workspace: Path
) -> tuple[list[BenchmarkSample], dict[tuple[Path, int], str]]:
    """Prepare PDF samples sequentially and cache extracted images."""
    extracted_dir = workspace / "extracted"
    extracted_dir.mkdir(parents=True)
    pdf_dir = workspace / "pdfs"
    pdf_dir.mkdir()
    samples: list[BenchmarkSample] = []
    failures: dict[tuple[Path, int], str] = {}

    jobs = (
        SampleJob(image_number, image_path, scale, workspace)
        for image_number, image_path in enumerate(image_paths)
        for scale in scales
    )
    for job in tqdm(
        jobs,
        total=len(image_paths) * len(scales),
        desc="Preparing PDF samples",
        unit="image",
    ):
        result = _prepare_one_sample(job)
        if isinstance(result, BenchmarkFailure):
            failures[(result.image_path, result.scale)] = result.error
            tqdm.write(f"Skipping unreadable image {result.image_path}: {result.error}")
            continue
        samples.append(result)

    samples.sort(key=lambda sample: (sample.image_path.name, sample.scale))
    return samples, failures


def test_prepare_samples_records_unreadable_images(tmp_path: Path) -> None:
    image_path = tmp_path / "truncated.png"
    encoded_image = BytesIO()
    Image.new("RGB", (256, 256), color="red").save(encoded_image, format="PNG")
    image_data = encoded_image.getvalue()
    image_path.write_bytes(image_data[: len(image_data) // 2])

    samples, failures = _prepare_samples([image_path], [2, 4], tmp_path / "work")

    assert not samples
    assert set(failures) == {(image_path, 2), (image_path, 4)}
    assert all("truncated" in message.lower() for message in failures.values())


def _reconstruction_metrics(reference: Image.Image, result: bytes) -> tuple[float, float, float]:
    """Return MSE, MAE, and PSNR after matching the source dimensions."""
    with Image.open(BytesIO(result)) as enhanced:
        reconstructed = enhanced.convert("RGB")
        if reconstructed.size != reference.size:
            reconstructed = reconstructed.resize(reference.size, Image.Resampling.LANCZOS)

        reference_pixels = np.asarray(reference, dtype=np.float64)
        reconstructed_pixels = np.asarray(reconstructed, dtype=np.float64)

    difference = reconstructed_pixels - reference_pixels
    mse = float(np.mean(np.square(difference)))
    mae = float(np.mean(np.abs(difference)))
    psnr = math.inf if mse == 0 else 10 * math.log10((255**2) / mse)
    return mse, mae, psnr


@pytest.mark.benchmark
def test_restoration_benchmark(request) -> None:
    """Write one error-metric row per image, scale, and enhancement method."""
    if not request.config.getoption("--run-benchmark"):
        pytest.skip("pass --run-benchmark to run this image benchmark")

    dataset_dir: Path = request.config.getoption("--benchmark-images-dir")
    image_paths = sorted(
        path
        for path in dataset_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if not image_paths:
        pytest.fail(f"No supported images found in {dataset_dir}")

    image_limit = request.config.getoption("--benchmark-limit")
    if image_limit < 0:
        raise pytest.UsageError("--benchmark-limit cannot be negative")
    if image_limit:
        image_paths = image_paths[:image_limit]

    scales = _parse_scales(request.config.getoption("--benchmark-scales"))
    models = _parse_models(request.config.getoption("--benchmark-models"))
    device = request.config.getoption("--benchmark-device")

    if models:
        import torch

        torch.set_num_threads(1)

    output_path: Path = request.config.getoption("--benchmark-csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    configurations = [
        (EnhancementMethod.LANCZOS, "lanczos"),
        *((EnhancementMethod.SUPER_IMAGE, model) for model in models),
    ]

    with TemporaryDirectory(prefix="benchmark-", dir=output_path.parent) as workspace:
        samples, preparation_failures = _prepare_samples(image_paths, scales, Path(workspace))
        samples_by_key = {(sample.image_path, sample.scale): sample for sample in samples}

        with output_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=(
                    "image",
                    "scale",
                    "method",
                    "model",
                    "width",
                    "height",
                    "mse",
                    "mae",
                    "psnr",
                    "error",
                ),
            )
            writer.writeheader()

            with tqdm(
                total=len(image_paths) * len(scales) * len(configurations),
                desc="Benchmarking",
                unit="image",
            ) as progress:
                for scale in scales:
                    for method, model_name in configurations:
                        options = EnhancementOptions(
                            method=method,
                            scale=float(scale),
                            model_name=model_name,
                            device=device,
                        )
                        progress.set_description(f"{scale}x {model_name}")
                        configuration_failure: str | None = None
                        for image_path in image_paths:
                            sample = samples_by_key.get((image_path, scale))
                            if sample is None:
                                row: dict[str, object] = {
                                    "image": image_path.name,
                                    "scale": scale,
                                    "method": method.value,
                                    "model": model_name,
                                    "width": "",
                                    "height": "",
                                    "mse": "",
                                    "mae": "",
                                    "psnr": "",
                                    "error": preparation_failures[(image_path, scale)],
                                }
                            elif configuration_failure is not None:
                                row = {
                                    "image": image_path.name,
                                    "scale": scale,
                                    "method": method.value,
                                    "model": model_name,
                                    "width": sample.width,
                                    "height": sample.height,
                                    "mse": "",
                                    "mae": "",
                                    "psnr": "",
                                    "error": configuration_failure,
                                }
                            else:
                                try:
                                    enhanced = enhance_image(
                                        sample.extracted_path.read_bytes(),
                                        options,
                                    )
                                except EnhancementOutOfMemoryError:
                                    raise
                                except EnhancementError as error:
                                    configuration_failure = str(error)
                                    row = {
                                        "image": image_path.name,
                                        "scale": scale,
                                        "method": method.value,
                                        "model": model_name,
                                        "width": sample.width,
                                        "height": sample.height,
                                        "mse": "",
                                        "mae": "",
                                        "psnr": "",
                                        "error": configuration_failure,
                                    }
                                else:
                                    with Image.open(sample.image_path) as source:
                                        reference = source.convert("RGB")
                                    mse, mae, psnr = _reconstruction_metrics(
                                        reference, enhanced
                                    )
                                    row = {
                                        "image": sample.image_path.name,
                                        "scale": scale,
                                        "method": method.value,
                                        "model": model_name,
                                        "width": sample.width,
                                        "height": sample.height,
                                        "mse": mse,
                                        "mae": mae,
                                        "psnr": psnr,
                                        "error": "",
                                    }
                            writer.writerow(row)
                            csv_file.flush()
                            progress.update()
