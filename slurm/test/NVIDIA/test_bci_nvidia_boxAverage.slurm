#!/bin/bash
#SBATCH --job-name=resshift_bci_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=4:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21213/projects/resshift/logs/test_bci.%j.out
#SBATCH -e /data/antwerpen/212/vsc21213/projects/resshift/logs/test_bci.%j.err

set -euo pipefail

# Set model.ckpt_path in configs/staining_bci.yaml to the EMA checkpoint before submitting.
# e.g. ckpt_path: /data/antwerpen/212/vsc21213/projects/resshift/outputs/experiments/bci/ema_ckpts/ema_model_best.pth

export PROJECT_NAME="ResShift"
export VSC_REPO="$VSC_DATA/projects/resshift/code/$PROJECT_NAME"
export CONTAINER="$VSC_SCRATCH/containers/resshift_nvidia.sif"
export DATA_SQSH="$VSC_SCRATCH/datasets/BCI_dataset.sqsh"
export DATA_DIR="$VSC_SCRATCH/datasets/BCI"
export SAVE_DIR="$VSC_DATA/projects/resshift/outputs/experiments/bci"
export INFER_OUT="$VSC_DATA/projects/resshift/outputs/inference/bci"
export PLOTS_DIR="$VSC_DATA/projects/resshift/outputs/plots"
export EVAL_RESULTS="$VSC_DATA/projects/resshift/outputs/eval_results.csv"

module purge
module load calcua/2025a

echo "Starting ResShift BCI Test on NVIDIA GPU..."
echo "Job ID: $SLURM_JOB_ID"

mkdir -p "$DATA_DIR"
mkdir -p "$INFER_OUT"
mkdir -p "$PLOTS_DIR"

unset SLURM_CPUS_PER_TASK

# 1. Run inference on BCI test set (HE → IHC)
srun apptainer exec --nv \
    -B "$DATA_SQSH:$DATA_DIR:image-src=/" \
    -B "$VSC_DATA:$VSC_DATA" \
    "$CONTAINER" \
    bash -c "cd $VSC_REPO && python inference_resshift.py \
        --task staining_bci \
        --scale 1 \
        -i $DATA_DIR/HE/test \
        -o $INFER_OUT"

echo ""
echo "Inference complete. Running benchmark evaluation..."

# 2. Evaluate predictions vs ground truth
srun apptainer exec --nv \
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

echo ""
echo "Plotting learning curves..."

# 3. Plot training loss + validation PSNR from training.log
srun apptainer exec \
    -B "$VSC_DATA:$VSC_DATA" \
    "$CONTAINER" \
    bash -c "cd $VSC_REPO && python plot_learning_curves_resshift.py \
        --exp $SAVE_DIR \
        --out $PLOTS_DIR \
        --name resshift_bci"

echo ""
echo "All done."
echo "Eval results   : $EVAL_RESULTS"
echo "Learning curves : $PLOTS_DIR/resshift_bci_learning_curves.png"
