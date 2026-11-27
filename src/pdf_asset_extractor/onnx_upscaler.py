"""ONNX Runtime super-resolution engine with DirectML/CUDA/CPU and tiled inference."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
import math
import os
from pathlib import Path
import ssl
import sys
from typing import Any, Final
import urllib.error
import urllib.request

import numpy as np

SUPPORTED_MODELS: Final[tuple[str, ...]] = ("drln-bam", "edsr", "msrn")
SUPPORTED_SCALES: Final[frozenset[int]] = frozenset({2, 4})
DEFAULT_MODELS_BASE_URL: Final[str] = (
    "https://github.com/gwiazdan/pdf-asset-extractor/releases/download/models-v1"
)
DEFAULT_TILE_SIZE: Final[int] = 256
DEFAULT_TILE_OVERLAP: Final[int] = 16
DOWNLOAD_CHUNK_SIZE: Final[int] = 64 * 1024
DOWNLOAD_PROGRESS_STEP_BYTES: Final[int] = 256 * 1024
DOWNLOAD_TIMEOUT_SECONDS: Final[int] = 30

DownloadProgressCallback = Callable[[int, int | None], None]
TileProgressCallback = Callable[[int, int], None]
CancelCheck = Callable[[], bool]
TensorRunner = Callable[[np.ndarray], np.ndarray]


class OnnxUpscalerError(RuntimeError):
    """Raised when an ONNX super-resolution model cannot be downloaded or executed."""


class OnnxCancelledError(OnnxUpscalerError):
    """Raised when a model download or inference operation is cancelled."""


def default_app_cache_dir() -> Path:
    """Return the per-user writable cache directory for models and diagnostic logs."""
    if sys.platform == "win32":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data) / "pdf-asset-extractor"
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        return Path(xdg_cache) / "pdf-asset-extractor"
    return Path.home() / ".cache" / "pdf-asset-extractor"


def models_cache_dir() -> Path:
    """Return the directory where downloaded ONNX super-resolution models are cached."""
    override = os.environ.get("PDF_ASSET_EXTRACTOR_MODELS_DIR")
    if override:
        return Path(override)
    return default_app_cache_dir() / "models"


def model_filename(model_name: str, scale: int) -> str:
    """Return the canonical ONNX filename for a model and scale factor."""
    return f"{model_name}-{scale}x.onnx"


def model_download_url(model_name: str, scale: int) -> str:
    """Return the remote release URL for an ONNX model file."""
    base_url = os.environ.get("PDF_ASSET_EXTRACTOR_MODELS_URL", DEFAULT_MODELS_BASE_URL).rstrip(
        "/"
    )
    return f"{base_url}/{model_filename(model_name, scale)}"


def _check_cancelled(cancel_check: CancelCheck | None) -> None:
    if cancel_check is not None and cancel_check():
        raise OnnxCancelledError("Eksport został anulowany.")


def ensure_model_file(
    model_name: str,
    scale: int,
    *,
    progress_callback: DownloadProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
) -> Path:
    """Return the cached local ONNX path, downloading the model on first use."""
    if model_name not in SUPPORTED_MODELS:
        raise OnnxUpscalerError(f"Nieznany model super-image: {model_name}")
    if scale not in SUPPORTED_SCALES:
        raise OnnxUpscalerError("Modele super-image obsługują wyłącznie skale 2x i 4x.")

    cache_dir = models_cache_dir()
    filename = model_filename(model_name, scale)
    target_path = cache_dir / filename
    if target_path.is_file() and target_path.stat().st_size > 0:
        return target_path

    _check_cancelled(cancel_check)
    cache_dir.mkdir(parents=True, exist_ok=True)
    url = model_download_url(model_name, scale)
    temp_path = target_path.with_suffix(".onnx.tmp")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "pdf-asset-extractor"},
    )
    ssl_context = ssl.create_default_context()

    try:
        with (
            urllib.request.urlopen(
                request,
                timeout=DOWNLOAD_TIMEOUT_SECONDS,
                context=ssl_context,
            ) as response,
            temp_path.open("wb") as output_file,
        ):
            content_length_header = response.headers.get("Content-Length")
            total_bytes = (
                int(content_length_header)
                if content_length_header and content_length_header.isdigit()
                else None
            )
            downloaded = 0
            last_reported = 0
            if progress_callback is not None:
                progress_callback(0, total_bytes)

            while True:
                _check_cancelled(cancel_check)
                chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                output_file.write(chunk)
                downloaded += len(chunk)
                if progress_callback is not None and (
                    downloaded - last_reported >= DOWNLOAD_PROGRESS_STEP_BYTES
                    or (total_bytes is not None and downloaded >= total_bytes)
                ):
                    progress_callback(downloaded, total_bytes)
                    last_reported = downloaded

            if downloaded == 0:
                raise OnnxUpscalerError(f"Pobrany plik modelu '{filename}' jest pusty.")
            if progress_callback is not None and downloaded != last_reported:
                progress_callback(downloaded, total_bytes)

        temp_path.replace(target_path)
        return target_path
    except OnnxCancelledError:
        temp_path.unlink(missing_ok=True)
        raise
    except urllib.error.HTTPError as error:
        temp_path.unlink(missing_ok=True)
        if error.code == 404:
            raise OnnxUpscalerError(
                f"Nie znaleziono pliku modelu '{filename}' pod adresem {url} (HTTP 404). "
                f"Uruchom workflow 'Export ONNX Models' w GitHub Actions lub umieść plik w "
                f"{cache_dir}."
            ) from error
        raise OnnxUpscalerError(
            f"Błąd pobierania modelu '{filename}' (HTTP {error.code}): {error.reason}"
        ) from error
    except (OSError, urllib.error.URLError) as error:
        temp_path.unlink(missing_ok=True)
        raise OnnxUpscalerError(f"Nie udało się pobrać modelu '{filename}': {error}") from error
    except Exception as error:
        temp_path.unlink(missing_ok=True)
        raise OnnxUpscalerError(f"Nie udało się zapisać modelu '{filename}': {error}") from error


def _provider_display_name(providers: Sequence[str]) -> str:
    if "DmlExecutionProvider" in providers:
        return "DirectML (GPU)"
    if "CUDAExecutionProvider" in providers:
        return "CUDA (GPU)"
    return "CPU"


def resolve_onnx_providers(
    requested_device: str,
    available_providers: Sequence[str] | None = None,
) -> tuple[list[str], str]:
    """Select ONNX Runtime execution providers and a human-readable accelerator label."""
    if available_providers is None:
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise OnnxUpscalerError(
                "Brak biblioteki onnxruntime. Zainstaluj zależności poleceniem: uv sync."
            ) from error
        available = list(ort.get_available_providers())
    else:
        available = list(available_providers)

    normalized = requested_device.strip().lower()
    if normalized == "cpu":
        return ["CPUExecutionProvider"], "CPU"
    if normalized == "auto":
        if "DmlExecutionProvider" in available:
            return ["DmlExecutionProvider", "CPUExecutionProvider"], "DirectML (GPU)"
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"], "CUDA (GPU)"
        return ["CPUExecutionProvider"], "CPU"
    if normalized in {"gpu", "dml"}:
        if "DmlExecutionProvider" in available:
            return ["DmlExecutionProvider", "CPUExecutionProvider"], "DirectML (GPU)"
        if normalized == "gpu" and "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"], "CUDA (GPU)"
        raise OnnxUpscalerError(
            "Wybrano akcelerację GPU, ale ONNX Runtime nie wykrywa DirectML ani CUDA."
        )
    if normalized == "cuda":
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"], "CUDA (GPU)"
        raise OnnxUpscalerError("Wybrano CUDA, ale ONNX Runtime nie wykrywa GPU CUDA.")
    raise OnnxUpscalerError(f"Nieznane urządzenie obliczeniowe: {requested_device}")


def detect_default_accelerator() -> str:
    """Return the label of the default ONNX Runtime accelerator on this machine."""
    try:
        _, label = resolve_onnx_providers("auto")
    except (OnnxUpscalerError, RuntimeError):
        return "CPU"
    return label


@dataclass(frozen=True, slots=True)
class OnnxUpscalerSession:
    """Loaded ONNX Runtime session ready to upscale float32 NCHW tensors."""

    session: Any
    input_name: str
    output_name: str
    scale: int
    accelerator_label: str

    def __call__(self, tensor: np.ndarray) -> np.ndarray:
        outputs = self.session.run([self.output_name], {self.input_name: tensor})
        return np.asarray(outputs[0], dtype=np.float32)


@lru_cache(maxsize=1)
def _load_cached_session(
    model_path: str,
    providers: tuple[str, ...],
    scale: int,
) -> OnnxUpscalerSession:
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise OnnxUpscalerError(
            "Brak biblioteki onnxruntime. Zainstaluj zależności poleceniem: uv sync."
        ) from error

    try:
        session_options = ort.SessionOptions()
        if "DmlExecutionProvider" in providers:
            # DirectML requires disabling memory pattern and parallel exec for dynamic sizes.
            session_options.enable_mem_pattern = False
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        session = ort.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=list(providers),
        )
        active_providers = session.get_providers()
        return OnnxUpscalerSession(
            session=session,
            input_name=session.get_inputs()[0].name,
            output_name=session.get_outputs()[0].name,
            scale=scale,
            accelerator_label=_provider_display_name(active_providers),
        )
    except Exception as error:
        raise OnnxUpscalerError(f"Nie udało się uruchomić sesji ONNX Runtime: {error}") from error


def clear_session_cache() -> None:
    """Release the cached ONNX Runtime inference session."""
    _load_cached_session.cache_clear()


def load_onnx_upscaler(
    model_name: str,
    scale: int,
    requested_device: str,
    *,
    download_progress_callback: DownloadProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
) -> OnnxUpscalerSession:
    """Ensure the ONNX model is cached locally and return a ready inference session."""
    providers, _ = resolve_onnx_providers(requested_device)
    model_path = ensure_model_file(
        model_name,
        scale,
        progress_callback=download_progress_callback,
        cancel_check=cancel_check,
    )
    _check_cancelled(cancel_check)
    return _load_cached_session(str(model_path), tuple(providers), scale)


def _rgb_to_nchw(rgb_pixels: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(
        rgb_pixels.astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...] / 255.0
    )


def _nchw_to_rgb_uint8(output_tensor: np.ndarray) -> np.ndarray:
    if output_tensor.ndim != 4 or output_tensor.shape[0] != 1 or output_tensor.shape[1] != 3:
        raise OnnxUpscalerError(
            f"Model ONNX zwrócił nieprawidłowy kształt tensora: {output_tensor.shape}"
        )
    clamped = np.clip(output_tensor[0], 0.0, 1.0).transpose(1, 2, 0)
    return np.rint(clamped * 255.0).astype(np.uint8)


def upscale_rgb_array(
    rgb_pixels: np.ndarray,
    *,
    runner: TensorRunner,
    scale: int,
    tile_size: int = DEFAULT_TILE_SIZE,
    tile_overlap: int = DEFAULT_TILE_OVERLAP,
    progress_callback: TileProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
) -> np.ndarray:
    """Upscale an HxWx3 uint8 RGB array using overlap-tile inference for large images."""
    if rgb_pixels.ndim != 3 or rgb_pixels.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB array, got shape {rgb_pixels.shape}")
    if scale <= 0:
        raise ValueError("scale must be greater than zero")
    if tile_size <= 0:
        raise ValueError("tile_size must be greater than zero")
    if tile_overlap < 0:
        raise ValueError("tile_overlap must be non-negative")

    height, width, _ = rgb_pixels.shape
    if height <= tile_size and width <= tile_size:
        _check_cancelled(cancel_check)
        if progress_callback is not None:
            progress_callback(1, 1)
        result = runner(_rgb_to_nchw(rgb_pixels))
        return _nchw_to_rgb_uint8(result)

    tiles_x = math.ceil(width / tile_size)
    tiles_y = math.ceil(height / tile_size)
    total_tiles = tiles_x * tiles_y
    output = np.empty((height * scale, width * scale, 3), dtype=np.uint8)

    for tile_y in range(tiles_y):
        for tile_x in range(tiles_x):
            _check_cancelled(cancel_check)
            tile_index = tile_y * tiles_x + tile_x + 1

            x0 = tile_x * tile_size
            x1 = min(x0 + tile_size, width)
            y0 = tile_y * tile_size
            y1 = min(y0 + tile_size, height)

            pad_x0 = max(x0 - tile_overlap, 0)
            pad_x1 = min(x1 + tile_overlap, width)
            pad_y0 = max(y0 - tile_overlap, 0)
            pad_y1 = min(y1 + tile_overlap, height)

            tile_rgb = rgb_pixels[pad_y0:pad_y1, pad_x0:pad_x1, :]
            tile_out_rgb = _nchw_to_rgb_uint8(runner(_rgb_to_nchw(tile_rgb)))

            crop_x0 = (x0 - pad_x0) * scale
            crop_x1 = (x1 - pad_x0) * scale
            crop_y0 = (y0 - pad_y0) * scale
            crop_y1 = (y1 - pad_y0) * scale

            output[y0 * scale : y1 * scale, x0 * scale : x1 * scale, :] = tile_out_rgb[
                crop_y0:crop_y1, crop_x0:crop_x1, :
            ]
            if progress_callback is not None:
                progress_callback(tile_index, total_tiles)

    return output
