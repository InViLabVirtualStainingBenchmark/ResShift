#!/bin/bash
#SBATCH --job-name=bci_overlap_nvidia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=7:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21213/projects/resshift/logs/test_bci_seamless_nvidia.%j.out
#SBATCH -e /data/antwerpen/212/vsc21213/projects/resshift/logs/test_bci_seamless_nvidia.%j.err

set -euo pipefail

export PROJECT_NAME="ResShift"
export VSC_REPO="$VSC_DATA/projects/resshift/code/$PROJECT_NAME"
export CONTAINER="$VSC_SCRATCH/containers/resshift_nvidia.sif"
export DATA_SQSH="$VSC_SCRATCH/datasets/BCI_dataset.sqsh"
export DATA_DIR="$VSC_SCRATCH/datasets/BCI"
export INFER_OUT="/scratch/antwerpen/grp/ap_invilab_td_thesis/diffusion-predictions/ResShift/BCI_Gaussian"

mkdir -p "$DATA_DIR"
mkdir -p "$INFER_OUT"

module purge
module load calcua/2025a

echo "Running Full BCI Seamless Inference on NVIDIA (Stride 448 + Gaussian Blending)..."
unset SLURM_CPUS_PER_TASK

srun apptainer exec --nv \
    -B "$DATA_SQSH:$DATA_DIR:image-src=/" \
    -B "$VSC_DATA:$VSC_DATA" \
    -B "/scratch/antwerpen/grp/ap_invilab_td_thesis:/scratch/antwerpen/grp/ap_invilab_td_thesis" \
    "$CONTAINER" \
    bash -c "cd $VSC_REPO && python inference_resshift.py \
        --task staining_bci \
        --scale 1 \
        --chop_size 512 \
        --chop_stride 448 \
        --blend_mode gaussian \
        -i $DATA_DIR/HE/test \
        -o $INFER_OUT"

echo "Full BCI Seamless Inference complete! Results at: $INFER_OUT"
