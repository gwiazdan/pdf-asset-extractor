# Workflow and Algorithms

## 1. End-to-End GUI User Flow

Launching `uv run gui` starts `MainWindow`:

1. **Folder selection (`FolderPickerWidget`)**
   - Opens at the user's home directory (`Path.home()`).
   - Users browse folders (double-click enters a folder, the **Up** button moves to the parent)
     and see `.pdf` files listed as read-only previews.
   - Confirming a selection emits the highlighted subdirectory (or current directory) and switches
     to the **Folder** view.
2. **PDF browsing**
   - The **Folder** view displays every `.pdf` file in the chosen directory as a tile with the
     custom `assets/pdf.svg` icon and wrapped filename.
   - Clicking a PDF starts an asynchronous `PDFExtractionJob` and switches to the **PDF** view.
3. **Subprocess image extraction**
   - `PDFExtractionJob` launches `extraction_worker.py` in a child process with a `60 s` timeout.
   - The worker opens the PDF with PyMuPDF, extracts every placed image across all pages, generates
     a PNG thumbnail (max side `256 px`), writes both to a temporary directory, and prints a JSON
     manifest to `stdout`.
   - The GUI reads the extracted payloads, removes the temporary directory, and populates the
     thumbnail grid with page number, image index, and original dimensions (`width x height px`).
4. **Enhancement configuration and batch export**
   - Users select one or more images (`ExtendedSelection`) and configure:
     - **Enhancement method**: *None*, *Lanczos*, *Unsharp Mask*, or *Super-image*;
     - **Resize mode** (for scaling methods): *Factor* (`1.5x`, `2x`, `4x` — `1.5x` is disabled
       for `Super-image`) or *Target width* (`1000 px`, `1500 px`, `2000 px`, `3000 px`);
     - **Model and device** (for `Super-image`): `drln-bam`, `edsr`, `msrn` and `Auto`, `CPU`,
       `CUDA`;
     - **Mask preset** (for *Unsharp Mask*): `1 / 100 / 0`, `2 / 150 / 3`, `3 / 200 / 5`
       (radius / percent / threshold);
     - **Output format**: *Original*, *PNG*, or *JPEG*.
   - After choosing an export folder, selected images are enhanced as a batch via
     `enhance_images(...)` (threaded on CPU or grouped by dimensions on `super-image`), converted
     with `encode_image(...)`, and saved as `page-XXXX-image-YYYY.<ext>`.

## 2. Image Enhancement Algorithms

### Manual Lanczos-3 Resampling (`image_processing.lanczos_resize`)

Resampling is implemented manually in NumPy as a separable 1D filter computing exact `float64`
kernel weights and accumulating into reusable `float32` buffers:

1. **Image mode normalization**:
   - `L`, `LA`, `RGB`, and `RGBA` modes are preserved natively.
   - Bilevel (`1`) images are converted to `L`; palette (`P`) and other modes (e.g., `CMYK`) are
     converted to `RGBA` (when transparency metadata is present) or `RGB` so palette indices are
     never interpolated directly.
2. **Premultiplied alpha**:
   - For `RGBA` and `LA` images, color channels are multiplied by $\alpha / 255$ prior to
     resampling and divided back by the resampled $\alpha / 255$ afterward (zeroing fully
     transparent pixels). This eliminates dark fringe halos along transparent edges.
3. **Lanczos kernel** (default $a = 3$):
   - Normalized sinc: $\operatorname{sinc}(x) = \frac{\sin(\pi x)}{\pi x}$ for $x \ne 0$ and $1$
     for $x = 0$.
   - Windowed kernel: $L_a(x) = \operatorname{sinc}(x) \cdot \operatorname{sinc}(x / a)$ for
     $|x| < a$ and $0$ outside its support.
4. **Pixel-center mapping and anti-aliasing**:
   - Each destination pixel $j$ maps to source center
     $c(j) = (j + 0.5) \cdot \frac{N_{\text{in}}}{N_{\text{out}}} - 0.5$.
   - When upscaling ($\text{ratio} \le 1$), the filter support radius is $a = 3$.
   - When downscaling ($\text{ratio} > 1$), the kernel is stretched by
     $\text{filter\_scale} = \text{ratio}$ for anti-aliasing. Out-of-bounds taps are zeroed and
     remaining in-bounds weights are normalized to sum to $1$.
5. **Optimal pass ordering**:
   - The algorithm automatically picks horizontal-first or vertical-first order to minimize the
     intermediate image pixel count, and skips any axis whose length does not change.
6. **Quantization**:
   - Results are rounded to the nearest integer (`np.rint`), clipped to $[0, 255]$, and cast to
     `uint8`.

### Manual Unsharp Mask (`image_processing.unsharp_mask`)

Sharpening implements a standard Gaussian **Unsharp Mask** without external filter libraries:

1. **Separable Gaussian blur**:
   - Builds a symmetric, normalized 1D kernel with half-width
     $r = \max(\lceil 3\sigma \rceil, 1)$ and weights $g(k) = \exp(-k^2 / (2\sigma^2))$ for
     $\sigma = \text{radius}$.
   - Convolves horizontally and vertically with edge-clamped padding (`mode="edge"`).
2. **Detail extraction and thresholding**:
   - High-frequency detail is computed as $D = I - B$ (original minus blurred image).
   - Pixels satisfying $|D| \ge \text{threshold}$ are updated as
     $O = I + \frac{\text{percent}}{100} \cdot D$; pixels below the threshold remain unchanged to
     avoid amplifying low-level noise in flat regions.
3. **Alpha preservation**:
   - For `RGBA` and `LA` images, only color/luminance channels are sharpened; the alpha channel is
     copied verbatim.
   - When `radius == 0` or `percent == 0`, an unmodified copy of the image is returned.

### Neural Super-Resolution (`image_enhancer._enhance_with_super_image_batch`)

1. Validates that the requested model scale is `2x`, `3x`, or `4x`.
2. Resolves the execution device (`auto` -> `cuda` if available, else `cpu`) and loads the model
   through the `@lru_cache(maxsize=1)` helper `_load_super_image_model` (downloading pretrained
   weights from Hugging Face on first use).
3. Converts inputs to `RGB`, groups images by `(width, height)` so heterogeneous selections can
   still run in batched tensors while preserving original order, and runs forward inference inside
   `torch.inference_mode()`.
4. Rounds output pixels to `uint8` and, when `ResizeMode.WIDTH` is active, resizes to the target
   width while preserving aspect ratio using the manual `lanczos_resize` function.

## 3. Export Format Encoding (`extractor.encode_image`)

- **`original`**:
  - When `EnhancementMethod.NONE` is active, writes the exact raw bytes and file extension
    extracted from the PDF.
  - When an enhancement method was applied, the enhanced pipeline outputs `PNG` bytes.
- **`png`**: converts non-grayscale/non-RGB colorspaces (such as CMYK) to RGB and encodes
  losslessly via `fitz.Pixmap.tobytes("png")`.
- **`jpg`**: strips any alpha channel, converts non-RGB colorspaces to 3-channel RGB, and encodes
  as JPEG.

## 4. Restoration Benchmark (`tests/benchmark.py`)

The opt-in benchmark measures end-to-end restoration quality across a full PDF round-trip:

1. Each source image in `images/` is downscaled by `2` and/or `4`, embedded into a temporary PDF,
   and extracted back through `extract_images_with_thumbnails`.
2. The extracted low-resolution image is passed to `enhance_image` for `lanczos` and each selected
   `super-image` model.
3. Reconstructed outputs are compared against the original image using:
   - **MSE** (Mean Squared Error),
   - **MAE** (Mean Absolute Error),
   - **PSNR** (Peak Signal-to-Noise Ratio in dB: $10 \log_{10}(255^2 / \text{MSE})$).
4. Per-image results are flushed incrementally to CSV (default `output/benchmark.csv`), recording
   unreadable images in the `error` column without aborting the run.
