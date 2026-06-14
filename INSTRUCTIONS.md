# ResShift: Master HPC Operations Guide (CalcUA)

This guide provides a definitive, phase-based workflow for deploying, training, and evaluating the ResShift Virtual Staining project on the CalcUA cluster.

---

## 1. Cluster Infrastructure & Key Paths

| Entity | Physical Cluster Path | Virtual Container Path |
| :--- | :--- | :--- |
| **Project Root** | `$VSC_DATA/projects/resshift` | `/data/antwerpen/.../resshift` |
| **Code Base** | `$VSC_DATA/projects/resshift/code/ResShift` | Executed from here |
| **BCI Dataset (.sqsh)** | `$VSC_SCRATCH/datasets/BCI_dataset.sqsh` | Mounted to `/tmp/BCI` |
| **MIST Dataset (.sqsh)** | `$VSC_SCRATCH/datasets/MIST_dataset.sqsh` | Mounted to `/tmp/MIST` |
| **Model Weights** | `$VSC_DATA/projects/resshift/outputs/experiments` | Mounts to `$SAVE_DIR` |
| **NVIDIA Container** | `$VSC_SCRATCH/containers/resshift_nvidia.sif` | -- |
| **AMD Container** | `$VSC_SCRATCH/containers/resshift_amd.sif` | -- |
| **Evaluation Container** | `$VSC_SCRATCH/containers/evaluate_amd.sif` | -- |

---

## Phase 1: Local Preparation

Before moving to the cluster, you must prepare your datasets and acquire the base weights.

### 1.1 Squash the Datasets (Local)
CalcUA requires datasets to be in **SquashFS** format for high-speed I/O.
1.  Organize your data locally: `BCI_dataset/HE/` and `BCI_dataset/IHC/`.
2.  Run the following command:
    ```bash
    mksquashfs BCI_dataset/ BCI_dataset.sqsh -noappend -processors 8
    ```

### 1.2 Acquire Base Weights (Local)
You must have the VQGAN weights before you can train.
*   **Weight Name:** `autoencoder_vq_f4.pth`
*   **Download:** [GitHub Release v2.0](https://github.com/zsyOAOA/ResShift/releases/download/v2.0/autoencoder_vq_f4.pth)

### 1.3 Upload to Cluster
```bash
# Upload Datasets to SCRATCH
scp BCI_dataset.sqsh vscXXXXX@login.hpc.uantwerpen.be:/scratch/antwerpen/212/vscXXXXX/datasets/

# Upload Base Weights directly to project folder on DATA
scp autoencoder_vq_f4.pth vscXXXXX@login.hpc.uantwerpen.be:/data/antwerpen/212/vscXXXXX/projects/resshift/code/ResShift/weights/
```

---

## Phase 2: Environment Setup (Cluster)

### 2.1 Build the Apptainer Containers
Build the `.sif` files from the provided `.def` files on a login node. Only build the Evaluation container if it isn't already present in your scratch space.

```bash
# Build NVIDIA Container (for ampere_gpu nodes)
apptainer build --fakeroot $VSC_SCRATCH/containers/resshift_nvidia.sif resshift_hpc.def

# Build AMD Container (for arcturus_gpu nodes)
apptainer build --fakeroot $VSC_SCRATCH/containers/resshift_amd.sif resshift_amd.def

# Build Evaluation Container (Optional - if not already on cluster)
apptainer build --fakeroot $VSC_SCRATCH/containers/evaluate_amd.sif evaluate_amd.def
```

### 2.2 Initialize Directory Structure
Run these scripts once to set up the standardized VSC folders and clone the necessary benchmark repositories.
```bash
cd $VSC_DATA/projects/resshift/code/ResShift/hpc
bash setup_project_resshift.sh
bash clone_repo_resshift.sh

# Clone Evaluation Repo (Optional - if not already in $VSC_DATA)
bash clone_repo_evaluate.sh
```

---

## Phase 3: Training Lifecycle

You can train on either **NVIDIA** or **AMD** hardware depending on queue availability.

### 3.1 Fresh Training Launch
**BCI Commands:**
```bash
# NVIDIA
sbatch slurm/train/NVIDIA/train_bci_nvidia.slurm
# AMD
sbatch slurm/train/AMD/train_bci_amd.slurm
```

**MIST Commands (Stain specific):**
*Add `--export=ALL,STAIN=<MARKER>` to specify which IHC marker to train.*
```bash
# ER Stain (NVIDIA or AMD)
sbatch --export=ALL,STAIN=ER slurm/train/NVIDIA/train_mist_nvidia.slurm
sbatch --export=ALL,STAIN=ER slurm/train/AMD/train_mist_amd.slurm

# PR / HER2 / Ki67 Stains (Example)
sbatch --export=ALL,STAIN=PR slurm/train/NVIDIA/train_mist_nvidia.slurm
```

### 3.2 Resuming Interrupted Training
If a job times out, follow this **Resumption Protocol**:

1.  **Check Logs:** Identify your latest checkpoint in `outputs/experiments/<task>/<timestamp>/ckpts/model_latest.pth`.
2.  **Submit with Resume Flag:**
    ```bash
    sbatch --export=ALL,RESUME_CKPT=/full/path/to/ckpts/model_latest.pth slurm/train/...
    ```

---

## Phase 4: Testing & High-Fidelity Inference

Inference translates H&E tiles into digital IHC stains. 

### 4.1 Mandatory Pre-Flight Configuration
Before running inference, you MUST point the configuration to your trained weights.
1.  Open your task config (e.g., `configs/staining_bci.yaml`).
2.  Update the `model.params.ckpt_path` to point to your **EMA** best model:
    ```yaml
    model:
      params:
        ckpt_path: "/data/antwerpen/.../ema_ckpts/ema_model_best.pth"
    ```

### 4.2 Executing Inference
Use the **GaussianBlend** scripts for publication-quality seamless images.

**BCI Commands:**
```bash
# Seamless (NVIDIA or AMD)
sbatch slurm/test/NVIDIA/test_bci_nvidia_GaussianBlend.slurm
sbatch slurm/test/AMD/test_bci_amd_GaussianBlend.slurm
```

**MIST Commands:**
```bash
# ER Stain (Seamless)
sbatch --export=ALL,STAIN=ER slurm/test/NVIDIA/test_mist_nvidia_GaussianBlend.slurm
```

### 4.3 Resume Inference
If an inference job stops, simply **resubmit the same command**. The script automatically skips existing result images in the output folder.

---

## Phase 5: Evaluation & Metrics

This generates final PSNR, SSIM, and LPIPS scores.

### 5.1 Verification
Ensure `PRED_DIR` in the eval script matches the `INFER_OUT` directory used in Phase 4.

### 5.2 Execution
```bash
# NVIDIA BCI Eval
sbatch slurm/eval/NVIDIA/eval_bci_nvidia_GaussianBlend.slurm

# MIST ER Eval
sbatch --export=ALL,STAIN=ER slurm/eval/NVIDIA/eval_mist_nvidia_GaussianBlend.slurm
```
**Results:** Check `$VSC_DATA/projects/resshift/outputs/eval_results.csv` for the final benchmark scores.

---

## Phase 6: Monitoring & Maintenance

*   **Queue Status:** `squeue -u $USER`
*   **Live Training Logs:** `tail -f $VSC_DATA/projects/resshift/logs/train_bci.<JOB_ID>.out`
*   **Path Troubleshooting:** If scripts crash with "File Not Found", verify the `-B` (Bind Mount) lines in your SLURM script match your physical directory structure on SCRATCH.
