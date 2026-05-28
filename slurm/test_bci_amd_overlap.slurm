#!/bin/bash
#SBATCH --job-name=bci_overlap_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=1:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p arcturus_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21213/projects/resshift/logs/test_bci_overlap.%j.out
#SBATCH -e /data/antwerpen/212/vsc21213/projects/resshift/logs/test_bci_overlap.%j.err

set -euo pipefail

export PROJECT_NAME="ResShift"
export VSC_REPO="$VSC_DATA/projects/resshift/code/$PROJECT_NAME"
export CONTAINER="$VSC_SCRATCH/containers/resshift_amd.sif"
export DATA_SQSH="$VSC_SCRATCH/datasets/BCI_dataset.sqsh"
export DATA_DIR="$VSC_SCRATCH/datasets/BCI"

# Save the test results in a specific subfolder so it doesn't overwrite your main inference
export INFER_OUT="/scratch/antwerpen/grp/ap_invilab_td_thesis/diffusion-predictions/ResShift/BCI_TEST"

# MIOPEN cache redirection for AMD GPUs
export MIOPEN_USER_DB_PATH=/tmp/miopen_$SLURM_JOB_ID
export MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen_$SLURM_JOB_ID
mkdir -p $MIOPEN_USER_DB_PATH

# Create a temporary directory on the compute node just for these 30 images
export TEMP_IN_DIR="/tmp/bci_test_30_$SLURM_JOB_ID"
mkdir -p "$TEMP_IN_DIR"
mkdir -p "$INFER_OUT"

module purge
module load calcua/2025a

echo "Extracting exactly 30 images from the SquashFS dataset for testing..."
unset SLURM_CPUS_PER_TASK

# Use the container to mount the dataset and copy exactly 30 images to our temp folder
srun apptainer exec \
    -B "$DATA_SQSH:$DATA_DIR:image-src=/" \
    -B "/tmp:/tmp" \
    "$CONTAINER" \
    bash -c "ls $DATA_DIR/HE/test/*.png | head -n 30 | xargs -I {} cp {} $TEMP_IN_DIR/"

echo "30 images copied. Running inference with --chop_stride 448 (64px overlap + Gaussian Blending)..."

# Run inference ONLY on the temporary folder containing the 30 images
srun apptainer exec --rocm \
    -B "/tmp:/tmp" \
    -B "$VSC_DATA:$VSC_DATA" \
    -B "/scratch/antwerpen/grp/ap_invilab_td_thesis:/scratch/antwerpen/grp/ap_invilab_td_thesis" \
    "$CONTAINER" \
    bash -c "cd $VSC_REPO && python inference_resshift.py \
        --task staining_bci \
        --scale 1 \
        --chop_size 512 \
        --chop_stride 448 \
        -i $TEMP_IN_DIR \
        -o $INFER_OUT"

# Clean up
rm -rf $MIOPEN_USER_DB_PATH
rm -rf $TEMP_IN_DIR

echo "Test complete! You can view the 30 results at: $INFER_OUT"