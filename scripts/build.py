"""Cross-platform PyInstaller build, smoke-test, and archive script."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import tomllib
import zipfile

ROOT_DIR = Path(__file__).resolve().parent.parent
APP_NAME = "pdf-asset-extractor"
SPEC_PATH = ROOT_DIR / "packaging" / f"{APP_NAME}.spec"
DIST_APP_DIR = ROOT_DIR / "dist" / APP_NAME
DEFAULT_OUTPUT_DIR = ROOT_DIR / "dist" / "artifacts"
WINDOWS_ICON_PATH = ROOT_DIR / "build" / "app.ico"


def _read_project_version() -> str:
    pyproject_path = ROOT_DIR / "pyproject.toml"
    with pyproject_path.open("rb") as handle:
        data = tomllib.load(handle)
    return str(data["project"]["version"])


def _normalize_version(raw_version: str | None) -> str:
    version = (raw_version or _read_project_version()).strip()
    if version.startswith("v") and len(version) > 1 and version[1].isdigit():
        version = version[1:]
    if not version:
        raise ValueError("Version must not be empty.")
    return version


def _platform_tag() -> str:
    machine = platform.machine().lower()
    arch = "x64" if machine in {"x86_64", "amd64"} else machine
    if sys.platform == "win32":
        return f"windows-{arch}"
    if sys.platform.startswith("linux"):
        return f"linux-{arch}"
    if sys.platform == "darwin":
        return f"macos-{arch}"
    return f"{sys.platform}-{arch}"


def _verify_cpu_torch_environment() -> None:
    """Ensure the Linux build environment uses CPU-only PyTorch without CUDA packages."""
    if not sys.platform.startswith("linux"):
        return

    import torch

    if "+cpu" not in torch.__version__:
        raise RuntimeError(
            f"Expected CPU-only PyTorch on Linux build, found torch=={torch.__version__}."
        )

    forbidden_prefixes = ("nvidia-", "cuda-", "triton")
    forbidden_dists = sorted(
        dist.metadata["Name"]
        for dist in metadata.distributions()
        if (name := dist.metadata.get("Name")) is not None
        and name.lower().startswith(forbidden_prefixes)
    )
    if forbidden_dists:
        raise RuntimeError(
            "Found CUDA/Triton packages in Linux CPU build environment: "
            + ", ".join(forbidden_dists)
        )


def _generate_windows_icon(svg_path: Path, ico_path: Path) -> None:
    """Rasterize the SVG logo into a multi-resolution Windows .ico file."""
    from PIL import Image
    import pymupdf as fitz

    ico_path.parent.mkdir(parents=True, exist_ok=True)
    with fitz.open(svg_path) as document:
        page = document[0]
        rect = page.rect
        scale = 256.0 / max(rect.width, rect.height, 1.0)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=True)
        image = Image.frombytes("RGBA", (pixmap.width, pixmap.height), pixmap.samples)

    sizes = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    image.save(ico_path, format="ICO", sizes=sizes)


def _binary_path(dist_dir: Path) -> Path:
    suffix = ".exe" if sys.platform == "win32" else ""
    return dist_dir / f"{APP_NAME}{suffix}"


def _run_pyinstaller() -> None:
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        str(SPEC_PATH),
    ]
    subprocess.run(command, cwd=ROOT_DIR, check=True)


def _run_smoke_test(binary: Path) -> None:
    if not binary.is_file():
        raise FileNotFoundError(f"Built binary not found: {binary}")

    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    completed = subprocess.run(
        [str(binary), "--self-check"],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Smoke test failed with exit code {completed.returncode}.\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )

    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("Smoke test produced no JSON output.")
    payload = json.loads(lines[-1])
    if (
        payload.get("status") != "ok"
        or payload.get("frozen") is not True
        or payload.get("extracted_images") != 1
    ):
        raise RuntimeError(f"Unexpected smoke test payload: {payload!r}")
    print(f"Smoke test passed: {payload}")


def _package_archive(dist_dir: Path, output_dir: Path, version: str, platform_tag: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"{APP_NAME}-{version}-{platform_tag}"
    if sys.platform == "win32":
        archive_path = output_dir / f"{base_name}.zip"
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for file_path in sorted(dist_dir.rglob("*")):
                if file_path.is_file():
                    arcname = Path(APP_NAME) / file_path.relative_to(dist_dir)
                    archive.write(file_path, arcname=str(arcname))
        return archive_path

    archive_path = output_dir / f"{base_name}.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(dist_dir, arcname=APP_NAME)
    return archive_path


def _write_sha256(archive_path: Path, platform_tag: str) -> tuple[Path, str]:
    digest = hashlib.sha256()
    with archive_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    checksum = digest.hexdigest()
    checksum_path = archive_path.parent / f"SHA256SUMS-{platform_tag}.txt"
    checksum_path.write_text(f"{checksum}  {archive_path.name}\n", encoding="utf-8")
    return checksum_path, checksum


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build and package PDF Asset Extractor.")
    parser.add_argument("--version", default=None, help="Release version override.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where packaged release archives are written.",
    )
    parser.add_argument(
        "--require-cpu-torch",
        action="store_true",
        help="Fail on Linux if CUDA/Triton packages or non-CPU PyTorch are installed.",
    )
    parser.add_argument(
        "--skip-smoke-test",
        action="store_true",
        help="Skip running --self-check on the built binary.",
    )
    arguments = parser.parse_args(argv)

    version = _normalize_version(arguments.version)
    platform_tag = _platform_tag()
    print(f"Building {APP_NAME} v{version} for {platform_tag}...")

    if arguments.require_cpu_torch:
        _verify_cpu_torch_environment()

    if sys.platform == "win32":
        _generate_windows_icon(ROOT_DIR / "assets" / "logo.svg", WINDOWS_ICON_PATH)

    _run_pyinstaller()

    binary = _binary_path(DIST_APP_DIR)
    if not arguments.skip_smoke_test:
        _run_smoke_test(binary)

    archive_path = _package_archive(DIST_APP_DIR, arguments.output_dir, version, platform_tag)
    checksum_path, checksum = _write_sha256(archive_path, platform_tag)
    size_mb = archive_path.stat().st_size / (1024 * 1024)
    print(f"Created archive: {archive_path} ({size_mb:.1f} MiB)")
    print(f"SHA256 ({checksum_path.name}): {checksum}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
