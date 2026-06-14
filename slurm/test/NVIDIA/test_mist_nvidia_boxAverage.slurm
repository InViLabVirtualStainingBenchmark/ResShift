#!/bin/bash
#SBATCH --job-name=resshift_mist_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=4:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21213/projects/resshift/logs/test_mist_%x.%j.out
#SBATCH -e /data/antwerpen/212/vsc21213/projects/resshift/logs/test_mist_%x.%j.err

set -euo pipefail

# Set model.ckpt_path in configs/staining_mist_<stain>.yaml to the EMA checkpoint before submitting.
# e.g. ckpt_path: /data/antwerpen/212/vsc21213/projects/resshift/outputs/experiments/mist_er/ema_ckpts/ema_model_best.pth
#
# Pass stain at submit time: sbatch --export=ALL,STAIN=ER slurm/test/test_mist_nvidia.slurm
if [ -z "${STAIN:-}" ]; then
    echo "ERROR: STAIN not set. Submit with: sbatch --export=ALL,STAIN=ER slurm/test/test_mist_nvidia.slurm"
    exit 1
fi

STAIN_LOWER=$(echo "$STAIN" | tr '[:upper:]' '[:lower:]')

export PROJECT_NAME="ResShift"
export VSC_REPO="$VSC_DATA/projects/resshift/code/$PROJECT_NAME"
export CONTAINER="$VSC_SCRATCH/containers/resshift_nvidia.sif"
export DATA_SQSH="$VSC_SCRATCH/datasets/MIST_dataset.sqsh"
export DATA_DIR="$VSC_SCRATCH/datasets/MIST"
export SAVE_DIR="$VSC_DATA/projects/resshift/outputs/experiments/mist_${STAIN_LOWER}"
export INFER_OUT="$VSC_DATA/projects/resshift/outputs/inference/mist_${STAIN_LOWER}"
export PLOTS_DIR="$VSC_DATA/projects/resshift/outputs/plots"
export EVAL_RESULTS="$VSC_DATA/projects/resshift/outputs/eval_results.csv"

module purge
module load calcua/2025a

echo "Starting ResShift MIST Test on NVIDIA GPU — Stain: $STAIN"
echo "Job ID: $SLURM_JOB_ID"

mkdir -p "$DATA_DIR"
mkdir -p "$INFER_OUT"
mkdir -p "$PLOTS_DIR"

unset SLURM_CPUS_PER_TASK

# 1. Run inference on MIST test set (valA → valB)
srun apptainer exec --nv \
    -B "$DATA_SQSH:$DATA_DIR:image-src=/" \
    -B "$VSC_DATA:$VSC_DATA" \
    "$CONTAINER" \
    bash -c "cd $VSC_REPO && python inference_resshift.py \
        --task staining_mist_${STAIN_LOWER} \
        --scale 1 \
        -i $DATA_DIR/$STAIN/TrainValAB/valA \
        -o $INFER_OUT"

echo ""
echo "Inference complete. Running benchmark evaluation..."

# 2. Evaluate predictions vs ground truth (valB is the MIST official test set)
srun apptainer exec --nv \
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

echo ""
echo "Plotting learning curves..."

# 3. Plot training loss + validation PSNR from training.log
srun apptainer exec \
    -B "$VSC_DATA:$VSC_DATA" \
    "$CONTAINER" \
    bash -c "cd $VSC_REPO && python plot_learning_curves_resshift.py \
        --exp $SAVE_DIR \
        --out $PLOTS_DIR \
        --name resshift_mist_${STAIN_LOWER}"

echo ""
echo "All done."
echo "Eval results   : $EVAL_RESULTS"
echo "Learning curves : $PLOTS_DIR/resshift_mist_${STAIN_LOWER}_learning_curves.png"
