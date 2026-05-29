#!/bin/bash
#SBATCH --job-name=eval_bci_overlap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=1:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p arcturus_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21213/projects/resshift/logs/eval_bci_overlap.%j.out
#SBATCH -e /data/antwerpen/212/vsc21213/projects/resshift/logs/eval_bci_overlap.%j.err

set -euo pipefail

# Paths
export CONTAINER="$VSC_SCRATCH/containers/evaluate_amd.sif"
export EVAL_REPO="$VSC_DATA/evaluate"

# BCI Overlap Test specific paths
export PRED_DIR="/scratch/antwerpen/grp/ap_invilab_td_thesis/diffusion-predictions/ResShift/BCI_TEST"
export DATA_SQSH="$VSC_SCRATCH/datasets/BCI_dataset.sqsh"
export DATA_DIR="/tmp/BCI_eval_overlap"
export OUTPUT_CSV="$VSC_DATA/projects/resshift/outputs/eval_results.csv"

module purge
module load calcua/2025a

echo "Starting BCI Overlap (Gaussian Blending) Evaluation on AMD..."
mkdir -p "$DATA_DIR"

export TEMP_GT_DIR="/tmp/BCI_eval_overlap_GT_30"
mkdir -p "$TEMP_GT_DIR"

# Copy ONLY the GT images that match the 30 predicted images
echo "Isolating the 30 matching Ground Truth images..."
srun apptainer exec \
    -B "$DATA_SQSH:$DATA_DIR:image-src=/" \
    -B "/tmp:/tmp" \
    -B "/scratch/antwerpen/grp/ap_invilab_td_thesis:/scratch/antwerpen/grp/ap_invilab_td_thesis" \
    "$CONTAINER" \
    bash -c "for file in $PRED_DIR/*.png; do cp \"$DATA_DIR/IHC/test/\$(basename \"\$file\")\" $TEMP_GT_DIR/ 2>/dev/null || true; done"


srun apptainer exec --rocm \
    -B "$DATA_SQSH:$DATA_DIR:image-src=/" \
    -B "$VSC_DATA:$VSC_DATA" \
    -B "/tmp:/tmp" \
    -B "/scratch/antwerpen/grp/ap_invilab_td_thesis:/scratch/antwerpen/grp/ap_invilab_td_thesis" \
    "$CONTAINER" \
    bash -c "cd $EVAL_REPO && python evaluate.py \
        --pred $PRED_DIR \
        --gt $TEMP_GT_DIR \
        --model_name resshift_bci_gaussian_fix \
        --dataset_name bci_30_subset \
        --split_name test \
        --match_by stem \
        --output $OUTPUT_CSV"

rm -rf $TEMP_GT_DIR
echo "BCI Overlap Evaluation Complete."