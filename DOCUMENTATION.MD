# Project Documentation: ResShift Virtual Staining Fixes

## 1. Environment & Testing Setup
- **Development Machine (Local):** Windows 11, Python 3.9.13, 4GB RTX 3050 Ti.
- **Verification Method:** Successful "Smoke Test" (5 iterations) on local machine to confirm code execution and initialization logic.
- **Target Machine:** Linux (WSL Ubuntu 24.04), Python 3.9.25, 11GB GTX 1080 Ti.

## 2. Identified Issues (The "Wrong" Results)
During the initial analysis, the project was found to have architectural misalignments:
1.  **Redundant Encoding conflict:** A logic conflict in the data pipeline was causing images to be compressed twice before reaching the diffusion layers. This destroyed pathology-specific cell details.
2.  **Config Misalignment:** The `image_size` was set to 256 in the UNet config, but the model was operating in a 64x64 latent space. This caused spatial misalignment in the Swin Transformer windows.
3.  **Inference Logic Error:** For restoration tasks (Scale 1), the `chop_size` was being multiplied by 4, causing it to bypass the tiling logic and lead to VRAM OOM on larger images.

## 3. Applied Changes

### A. Trainer/Data Logic Fixes
- **Fix:** Removed the redundant `.encode()` and `.detach()` calls in the data preparation pipeline.
- **Rationale:** The diffusion model’s core logic is already responsible for encoding images into the latent space. Removing the redundant step ensures the model sees the correct 64x64 latent resolution.

### B. Configuration Alignment (`configs/staining_bci.yaml`)
- **Change:** Updated `model.params.image_size` from `256` to `64`.
- **Change:** Kept `lq_size` at `256`.
- **Rationale:** In ResShift (LDM), `image_size` refers to the **latent resolution**. Since we use an `f4` autoencoder, a 256px image becomes a 64px latent. This change enables the learned Convolutional Feature Extractor to properly process the high-resolution HE input.

### C. Inference Optimization (`inference_resshift.py`)
- **Fix:** Updated the `chop_size` and `chop_stride` logic.
- **Rationale:** The multiplier `(4 // args.scale)` is now only applied if `scale > 1` (Super-Resolution). For Virtual Staining (`scale = 1`), it respects the user-defined `chop_size` (e.g., 256), ensuring tiling works correctly on memory-limited GPUs.

### D. Code Stability
- **Fix:** Added a `hasattr` check for metric initialization in the logging logic.
- **Rationale:** Fixed an `AttributeError` that occurred when logging frequencies were misaligned, ensuring the script always initializes properly.
