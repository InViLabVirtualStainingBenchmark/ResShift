# ResShift Virtual Staining: The Comprehensive Master Guide

This document is the definitive resource for the ResShift Virtual Staining adaptation. It combines the **technical rationale** (the "why") with a **practical deployment guide** (the "how").

---

## 1. Architectural Deep Dive: From SR to 1:1 Staining

The original ResShift was built for **Super-Resolution (SR)** (e.g., 64px $\to$ 256px). Virtual staining is a **1:1 Image-to-Image** task (e.g., 256px $\to$ 256px).

### A. The "Odd Resolution" & Alignment Problem
In the original code, the H&E conditioning image was downsampled using a hardcoded loop based on a fixed 4x ratio. If you used a non-standard resolution (like 512px), the math would fail, leading to a spatial mismatch and a fatal crash.

**Logic Transformation (`models/unet.py`):**

| **Component** | **Original Logic (SR-Assumed)** | **Modified Logic (Robust 1:1)** |
| :--- | :--- | :--- |
| **Downsample Loop** | `for ii in range(int(math.log(lq_size / image_size) / math.log(2))):` | `num_down = int(math.log(lq_size / image_size) / math.log(2))` |
| **Safety Net** | *None. Assumed dimensions matched.* | `if lq.shape[2:] != x.shape[2:]:`<br>`    lq = F.interpolate(lq, size=x.shape[2:], ...)` |

**The "Why":** This ensures that whether you use 256px, 512px, or 1024px tiles, the H&E guidance is always perfectly aligned with the latent noise, preventing "Out of Memory" or "Shape Mismatch" crashes.

---

## 2. Whole Slide Inference: Seamless Tiling

Whole Slide Images (WSI) are too large for GPU memory and must be processed in tiles.

### A. The Stride Correction
The original code multiplied the stride by 4 (assuming 4x upscaling). For virtual staining (`scale=1`), this caused the model to skip 75% of the image.

**Logic Transformation (`inference_resshift.py`):**

| **Feature** | **Original Code** | **Modified Code** |
| :--- | :--- | :--- |
| **Stride Math** | `stride = (size - 64) * (4 // scale)` | `if scale > 1: stride *= (4 // scale)` |
| **Effect** | Skips pixels if `scale=1`. | Precise 1:1 tile stepping. |

### B. Gaussian Blending (Removing Grid Lines)
Standard tiling leaves visible "Box" seams because diffusion is stochastic. We introduced **Gaussian Blending** (Hann Windowing) in `utils/util_image.py`.
*   **How it works:** Tiles overlap (e.g., 64px). Each tile is multiplied by a "bell-curve" mask that fades the edges to zero.
*   **Result:** Overlapping areas blend smoothly, creating a perfectly continuous stain with zero grid artifacts.

---

## 3. Training & Big Data Strategy

### A. Automated Validation Splitting
Pathology datasets are massive. We replaced manual partitioning with automated logic in `trainer.py`:
*   **Feature:** `val_split: 0.1` in the YAML.
*   **Benefit:** The system automatically carves out 10% of your data for validation using a fixed seed, ensuring reproducibility across all GPUs.

### B. The Rolling Checkpoint System
To save storage on HPC clusters, we use a three-tier system instead of iteration-based saves:
1.  **`model_latest.pth`**: Saved every validation cycle. Use this to **resume** a crashed/timed-out job.
2.  **`model_best.pth`**: Saved only when **PSNR improves**. This is your scientifically optimal model.
3.  **`model_last.pth`**: Saved when training hits 100%.

### C. EMA Weights: The Secret to Quality
Inside `ema_ckpts/`, you will find `ema_model_best.pth`. 
*   **What is it?** A slow-moving average of the weights.
*   **Why use it?** It filters out training noise. **Always use EMA weights for final inference** as they provide smoother, more realistic IHC textures.

---

## 4. Performance Metrics: PSNR & SSIM

We evaluate the model using **PSNR (Peak Signal-to-Noise Ratio)**.
*   **Is Higher Better?** **YES.** A higher PSNR means the AI's digital stain is a more faithful translation of the ground-truth IHC slide.
*   **Target:** We track PSNR during validation; the trainer only updates the `best` checkpoint when the PSNR increases.

---

## 5. Practical Guide: Step-by-Step Deployment

### Step 1: Environment Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements_frozen.txt
# Download weights/autoencoder_vq_f4.pth from GitHub Releases
```

### Step 2: Training
To start a standard training run (requires substantial VRAM):
`python main.py --cfg_path configs/staining_bci.yaml --save_dir ./save_dir/bci`

*For HPC users, it is recommended to use the SLURM scripts provided in the next phase.*

### Step 3: Inference (Production Quality)
To generate seamless, high-quality stains, use the Gaussian blend mode:
```bash
python inference_resshift.py \
    -i <input_dir> -o <output_dir> \
    --task staining_bci --scale 1 \
    --chop_size 512 --chop_stride 448 \
    --blend_mode gaussian
```

---

## 6. Visualization & Reporting: Training Curves

To visualize the model's progress and compare different benchmark runs, we use the `plot_training_curves_diffusion_models.py` script. This tool automatically parses the complex ResShift logs (including timestep-specific losses) into publication-quality graphs.

### A. How to generate graphs
Run the script from your project root:
```bash
python plot_training_curves_diffusion_models.py \
    --logs path/to/your/training.log \
    --labels ResShift-BCI \
    --name bci_results \
    --out-dir ./results
```

### B. Key Features
*   **Automatic Parsing:** Specifically designed to recognize ResShift's `t(1)`, `t(8)`, and `t(15)` loss formats.
*   **Best-Point Discovery:** Automatically identifies and stars the "Best" PSNR/LPIPS iteration with a callout box.
*   **Summary Harvesting:** Generates a `<name>_training_summary.csv` which extracts final metrics and total training time (HPC hours).

---

## 7. Architecture Preservation Audit
| Component | Status | Rationale |
| :--- | :--- | :--- |
| **Markov Chain Math** | **Untouched** | Preserves the core ResShift diffusion physics. |
| **VQGAN Autoencoder** | **Untouched** | Uses standard pre-trained weights to ensure latent consistency. |
| **Swin-UNet Layers** | **Untouched** | The internal transformer blocks are identical to the original paper. |

## 7. Current Status of the Project (June 2026)

As of the current benchmark phase, the ResShift models (BCI and MIST markers) have been trained but have not yet reached their full intended duration.

*   **Training Progress:** The models have reached approximately **55k iterations**.
*   **Original Target:** The ideal training duration is **300k iterations**. While a mid-term goal of 100k was set, cluster wait times and HPC time limits resulted in the current 55k state.
*   **Observations:** The results at 55k iterations are already looking **highly promising**. The digital IHC stains show strong structural alignment and marker-specific intensity.
*   **Recommendation:** To fully evaluate the "Residual Shift" capability and see the model's peak performance, it is recommended to **resume training** toward the 300k mark. This will allow for a definitive decision on whether further iterations yield significant qualitative improvements in cellular detail or if the model plateaus earlier.

**Conclusion:** This repository provides a robust, production-ready pipeline for Virtual Staining while remaining 100% faithful to the original ResShift architecture for benchmarking.
