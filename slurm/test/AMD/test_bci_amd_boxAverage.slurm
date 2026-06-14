#!/bin/bash
#SBATCH --job-name=resshift_bci_test_amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=4:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p arcturus_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21213/projects/resshift/logs/test_bci_amd.%j.out
#SBATCH -e /data/antwerpen/212/vsc21213/projects/resshift/logs/test_bci_amd.%j.err

set -euo pipefail

# Set model.ckpt_path in configs/staining_bci.yaml to the EMA checkpoint before submitting.
export PROJECT_NAME="ResShift"
export VSC_REPO="$VSC_DATA/projects/resshift/code/$PROJECT_NAME"
export CONTAINER="$VSC_SCRATCH/containers/resshift_amd.sif"
export DATA_SQSH="$VSC_SCRATCH/datasets/BCI_dataset.sqsh"
export DATA_DIR="$VSC_SCRATCH/datasets/BCI"
export SAVE_DIR="$VSC_DATA/projects/resshift/outputs/experiments/bci/2026-05-25-05-20"
export INFER_OUT="/scratch/antwerpen/grp/ap_invilab_td_thesis/diffusion-predictions/ResShift/BCI"
export PLOTS_DIR="$VSC_DATA/projects/resshift/outputs/plots"
export EVAL_RESULTS="$VSC_DATA/projects/resshift/outputs/eval_results.csv"

# MIOPEN cache redirection for AMD GPUs
export MIOPEN_USER_DB_PATH=/tmp/miopen_$SLURM_JOB_ID
export MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen_$SLURM_JOB_ID
mkdir -p $MIOPEN_USER_DB_PATH

module purge
module load calcua/2025a

echo "Starting ResShift BCI Test on AMD GPU..."
echo "Job ID: $SLURM_JOB_ID"

mkdir -p "$DATA_DIR"
mkdir -p "$INFER_OUT"
mkdir -p "$PLOTS_DIR"

unset SLURM_CPUS_PER_TASK

# 1. Run inference on BCI test set (HE → IHC)
srun apptainer exec --rocm \
    -B "$DATA_SQSH:$DATA_DIR:image-src=/" \
    -B "$VSC_DATA:$VSC_DATA" \
    "$CONTAINER" \
    bash -c "cd $VSC_REPO && python inference_resshift.py \
        --task staining_bci \
        --scale 1 \
        --blend_mode box \
        -i $DATA_DIR/HE/test \
        -o $INFER_OUT"

echo ""
echo "Inference complete. Running benchmark evaluation..."

# 2. Evaluate predictions vs ground truth
srun apptainer exec --rocm \
    -B "$DATA_SQSH:$DATA_DIR:image-src=/" \
    -B "$VSC_DATA:$VSC_DATA" \
    "$CONTAINER" \
    bash -c "python $VSC_REPO/Benchmark_evaluation/evaluate.py \
        --pred $INFER_OUT \
        --gt $DATA_DIR/IHC/test \
        --model_name resshift_bci \
        --dataset_name bci \
        --split_name test \
        --output $EVAL_RESULTS"

rm -rf $MIOPEN_USER_DB_PATH
echo "Cleanup complete. All done."
