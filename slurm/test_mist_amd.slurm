#!/bin/bash
#SBATCH --job-name=resshift_mist_test_amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=4:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p arcturus_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21213/projects/resshift/logs/test_mist_amd_%x.%j.out
#SBATCH -e /data/antwerpen/212/vsc21213/projects/resshift/logs/test_mist_amd_%x.%j.err

set -euo pipefail

# Submit with: sbatch --export=ALL,STAIN=ER slurm/test_mist_amd.slurm
if [ -z "${STAIN:-}" ]; then
    echo "ERROR: STAIN not set."
    exit 1
fi

STAIN_LOWER=$(echo "$STAIN" | tr '[:upper:]' '[:lower:]')

export PROJECT_NAME="ResShift"
export VSC_REPO="$VSC_DATA/projects/resshift/code/$PROJECT_NAME"
export CONTAINER="$VSC_SCRATCH/containers/resshift_amd.sif"
export DATA_SQSH="$VSC_SCRATCH/datasets/MIST_dataset.sqsh"
export DATA_DIR="$VSC_SCRATCH/datasets/MIST"
# We'll map the SAVE_DIR inside the conditional block below or use a dynamic approach
export INFER_OUT="/scratch/antwerpen/grp/ap_invilab_td_thesis/diffusion-predictions/ResShift/MIST_${STAIN}"
export EVAL_RESULTS="$VSC_DATA/projects/resshift/outputs/eval_results.csv"

# Set SAVE_DIR based on the specific timestamped folder provided by the user
if [ "$STAIN" == "ER" ]; then
    export SAVE_DIR="$VSC_DATA/projects/resshift/outputs/experiments/mist_er/2026-05-25-11-35"
elif [ "$STAIN" == "PR" ]; then
    export SAVE_DIR="$VSC_DATA/projects/resshift/outputs/experiments/mist_pr/2026-05-25-11-56"
elif [ "$STAIN" == "KI67" ]; then
    export SAVE_DIR="$VSC_DATA/projects/resshift/outputs/experiments/mist_he/2026-05-26-01-20"
else
    export SAVE_DIR="$VSC_DATA/projects/resshift/outputs/experiments/mist_${STAIN_LOWER}"
fi

# MIOPEN cache redirection
export MIOPEN_USER_DB_PATH=/tmp/miopen_$SLURM_JOB_ID
export MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen_$SLURM_JOB_ID
mkdir -p $MIOPEN_USER_DB_PATH

module purge
module load calcua/2025a

echo "Starting ResShift MIST Test (AMD) — Stain: $STAIN"

mkdir -p "$DATA_DIR"
mkdir -p "$INFER_OUT"

unset SLURM_CPUS_PER_TASK

srun apptainer exec --rocm \
    -B "$DATA_SQSH:$DATA_DIR:image-src=/" \
    -B "$VSC_DATA:$VSC_DATA" \
    "$CONTAINER" \
    bash -c "cd $VSC_REPO && python inference_resshift.py \
        --task staining_mist_${STAIN_LOWER} \
        --scale 1 \
        -i $DATA_DIR/$STAIN/TrainValAB/valA \
        -o $INFER_OUT"

srun apptainer exec --rocm \
    -B "$DATA_SQSH:$DATA_DIR:image-src=/" \
    -B "$VSC_DATA:$VSC_DATA" \
    "$CONTAINER" \
    bash -c "python $VSC_REPO/Benchmark_evaluation/evaluate.py \
        --pred $INFER_OUT \
        --gt $DATA_DIR/$STAIN/TrainValAB/valB \
        --model_name resshift_mist_${STAIN_LOWER} \
        --dataset_name mist \
        --split_name test \
        --output $EVAL_RESULTS"

rm -rf $MIOPEN_USER_DB_PATH
