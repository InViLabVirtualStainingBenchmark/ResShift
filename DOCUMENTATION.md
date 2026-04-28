# Project Documentation: ResShift Virtual Staining

This document serves as the master guide for anyone cloning this project. it details the setup, the architectural fixes applied to the original ResShift codebase, and the execution protocol for virtual staining.

## 1. Onboarding & Setup

### A. Environment Resolution
Ensure you are using the specific Python/PyTorch lock for this project. The verified dependencies are captured in `requirements_frozen.txt`:
```bash
# Install the exact verified environment
pip install -r requirements_frozen.txt
```

### B. Required Weight Downloads
You must download the pre-trained VQGAN autoencoder. This serves as the latent base for all staining tasks.
```bash
# Create weights directory
mkdir -p weights

# Download VQGAN base (f4 resolution)
wget https://github.com/zsyOAOA/ResShift/releases/download/v2.0/autoencoder_vq_f4.pth -O weights/autoencoder_vq_f4.pth
```

### C. Data Organization
The project is configured to point directly to the central source datasets. No local data handling is required.
- **BCI:** `/home/vs_user/Virtual Staining/Datasets/BCI/HE/train` and `test`
- **MIST:** `/home/vs_user/Virtual Staining/Datasets/MIST/[Marker]/TrainValAB/trainA` and `valA` (Markers: ER, PR, HER2, Ki67)

---

## 2. Applied Changes (Fixes to Original Codebase)

The following surgical changes were made to the original ResShift repository to enable high-fidelity pathology staining:

### A. Trainer/Data Logic Fixes
- **Fix:** Removed the redundant `.encode()` and `.detach()` calls in the data preparation pipeline.
- **Rationale:** The diffusion model’s core logic is already responsible for encoding images into the latent space. Removing the redundant step ensures the model sees the correct 64x64 latent resolution, preventing the destruction of pathology-specific cell details.

### B. Configuration Alignment (`configs/staining_bci.yaml`)
- **Change:** Updated `model.params.image_size` from `256` to `64`.
- **Change:** Kept `lq_size` at `256`.
- **Rationale:** In ResShift (LDM), `image_size` refers to the **latent resolution**. Since we use an `f4` autoencoder, a 256px image becomes a 64px latent. This change enables the learned Convolutional Feature Extractor to properly process the high-resolution input (H&E or Unstained) without spatial misalignment.

### C. Inference Optimization (`inference_resshift.py`)
- **Fix:** Updated the `chop_size` and `chop_stride` logic.
- **Rationale:** The multiplier `(4 // args.scale)` is now only applied if `scale > 1` (Super-Resolution). For Virtual Staining (`scale = 1`), it respects the user-defined `chop_size` (e.g., 256 or 512), ensuring tiling/chopping works correctly on memory-limited GPUs like the 1080 Ti.

### D. Code Stability
- **Fix:** Added a `hasattr` check for metric initialization in the logging logic.
- **Rationale:** Fixed an `AttributeError` that occurred when logging frequencies were misaligned, ensuring the script always initializes properly regardless of the iteration count.

---

## 3. Project Structure
To maintain a clean development environment, all outputs are redirected away from the root:
- **`configs/`**: Main production configurations (300k iterations).
- **`configs/smoke_test/`**: Rapid verification configs (50 iterations) for checking data/GPU.
- **`experiments/`**: All training outputs (checkpoints, logs, and visuals).
- **`results/`**: Final processed images from inference.

---

## 4. Execution Protocol

### Step 1: Verification (50-Iteration Smoke Test)
Run this to confirm the pipeline is working (~1 minute).
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 main.py --cfg_path configs/smoke_test/staining_bci.yaml --save_dir experiments/verification/bci
```

### Step 2: Production Training (300,000 Iterations)
Run the marker of your choice.
```bash
# BCI (H&E to IHC)
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 main.py --cfg_path configs/staining_bci.yaml --save_dir experiments/production/bci

# MIST Markers (Replace 'er' with 'pr', 'her2', or 'ki67')
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 main.py --cfg_path configs/staining_mist_er.yaml --save_dir experiments/production/er
```

### Step 3: Inference (Testing)
```bash
# BCI (H&E to IHC)
python inference_resshift.py -i "/home/vs_user/Virtual Staining/Datasets/BCI/HE/test" -o results/bci_val --task staining_bci --scale 1 --bs 1

# MIST (Example ER - Replace 'ER' and 'er' as needed)
python inference_resshift.py -i "/home/vs_user/Virtual Staining/Datasets/MIST/ER/TrainValAB/valA" -o results/mist_er --task staining_mist_er --scale 1 --bs 1
```
