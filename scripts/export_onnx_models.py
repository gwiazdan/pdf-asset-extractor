"""Export pretrained super-image PyTorch models to verified dynamic-shape ONNX files."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from pdf_asset_extractor.image_enhancer import SUPER_IMAGE_MODELS
from pdf_asset_extractor.onnx_upscaler import SUPPORTED_SCALES, model_filename

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = ROOT_DIR / "dist" / "models"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _export_one_model(
    model_name: str,
    model_class_name: str,
    scale: int,
    output_dir: Path,
    *,
    max_abs_tolerance: float = 1e-4,
) -> Path:
    import onnxruntime as ort
    import super_image  # pyright: ignore[reportMissingImports]
    import torch  # pyright: ignore[reportMissingImports]

    model_class: Any = getattr(super_image, model_class_name)
    model = model_class.from_pretrained(f"eugenesiow/{model_name}", scale=scale)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model = model.to("cpu").eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / model_filename(model_name, scale)
    dummy_input = torch.randn(1, 3, 64, 64, dtype=torch.float32)

    torch.onnx.export(
        model,
        (dummy_input,),
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "out_height", 3: "out_width"},
        },
        opset_version=17,
        dynamo=False,
    )

    # Verify numerical parity on a non-square shape different from the export dummy input.
    test_input = torch.rand(1, 3, 48, 80, dtype=torch.float32)
    with torch.inference_mode():
        pytorch_output = model(test_input).cpu().numpy()

    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    onnx_output = np.asarray(
        session.run(None, {"input": test_input.numpy()})[0],
        dtype=np.float32,
    )
    expected_shape = (1, 3, 48 * scale, 80 * scale)
    if onnx_output.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected ONNX output shape for {output_path.name}: "
            f"expected {expected_shape}, got {onnx_output.shape}"
        )

    max_diff = float(np.max(np.abs(pytorch_output - onnx_output)))
    if max_diff > max_abs_tolerance:
        raise RuntimeError(
            f"ONNX numerical drift too high for {output_path.name}: "
            f"max_diff={max_diff:.2e} > {max_abs_tolerance:.2e}"
        )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Exported {output_path.name}: {size_mb:.1f} MiB (max_diff={max_diff:.2e})")
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    model_map = dict(SUPER_IMAGE_MODELS)
    parser = argparse.ArgumentParser(description="Export super-image models to ONNX.")
    parser.add_argument(
        "--models",
        default=",".join(model_map),
        help="Comma-separated model names to export (default: all supported models).",
    )
    parser.add_argument(
        "--scales",
        default="2,4",
        help="Comma-separated integer scales to export (default: 2,4).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where exported .onnx files and SHA256SUMS.txt are written.",
    )
    arguments = parser.parse_args(argv)

    selected_models = [item.strip() for item in arguments.models.split(",") if item.strip()]
    selected_scales = [int(item.strip()) for item in arguments.scales.split(",") if item.strip()]

    unknown_models = [name for name in selected_models if name not in model_map]
    if unknown_models:
        raise SystemExit(f"Unknown model names: {unknown_models}")
    unsupported_scales = [scale for scale in selected_scales if scale not in SUPPORTED_SCALES]
    if unsupported_scales:
        raise SystemExit(f"Unsupported scales: {unsupported_scales}")

    exported_files: list[Path] = []
    for model_name in selected_models:
        for scale in selected_scales:
            exported_files.append(
                _export_one_model(model_name, model_map[model_name], scale, arguments.output_dir)
            )

    checksum_lines = [f"{_sha256(path)}  {path.name}" for path in sorted(exported_files)]
    checksum_path = arguments.output_dir / "SHA256SUMS.txt"
    checksum_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(f"Wrote checksums: {checksum_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
