#!/bin/bash
#SBATCH --job-name=resshift_bci
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=20:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1

set -euo pipefail

export PROJECT_NAME="ResShift"
export VSC_REPO="$VSC_DATA/projects/resshift/code/$PROJECT_NAME"
export CONTAINER="$VSC_SCRATCH/containers/resshift_hpc.sif"
export DATA_SQSH="$VSC_SCRATCH/datasets/BCI_dataset.sqsh"
export SAVE_DIR="$VSC_DATA/projects/resshift/outputs/experiments/bci"

mkdir -p "$SAVE_DIR"

module purge
module load calcua/2025a

echo "Starting ResShift BCI Training on HPC..."
echo "Job ID: $SLURM_JOB_ID"

srun apptainer exec --nv \
    -B "$DATA_SQSH:/data/BCI:image-src=/" \
    "$CONTAINER" \
    bash -c "cd $VSC_REPO && python main.py \
        --cfg_path configs/staining_bci.yaml \
        --save_dir $SAVE_DIR"

echo "Done! Results saved to $SAVE_DIR"
