#!/bin/bash
#SBATCH --job-name=resshift_mist
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
export DATA_SQSH="$VSC_SCRATCH/datasets/MIST_dataset.sqsh"

module purge
module load calcua/2025a

echo "Starting ResShift MIST Training on HPC..."
echo "Job ID: $SLURM_JOB_ID"

# MIST.sqsh contains ER/ HER2/ Ki67/ PR/ at its root.
for STAIN in ER HER2 Ki67 PR; do
    STAIN_LOWER=$(echo "$STAIN" | tr '[:upper:]' '[:lower:]')
    SAVE_DIR="$VSC_DATA/projects/resshift/outputs/experiments/mist_${STAIN_LOWER}"

    mkdir -p "$SAVE_DIR"

    echo ""
    echo "===== Stain: $STAIN ====="

    srun apptainer exec --nv \
        -B "$DATA_SQSH:/data/MIST:image-src=/" \
        "$CONTAINER" \
        bash -c "cd $VSC_REPO && python main.py \
            --cfg_path configs/staining_mist_${STAIN_LOWER}.yaml \
            --save_dir $SAVE_DIR"

    echo "  $STAIN done."
done

echo ""
echo "All stains complete."
