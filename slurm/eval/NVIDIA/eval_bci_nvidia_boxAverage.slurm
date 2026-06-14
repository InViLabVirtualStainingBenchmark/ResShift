#!/bin/bash
#SBATCH --job-name=eval_bci_nvidia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=10:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21213/projects/resshift/logs/eval_bci_nvidia.%j.out
#SBATCH -e /data/antwerpen/212/vsc21213/projects/resshift/logs/eval_bci_nvidia.%j.err

set -euo pipefail

# Paths - Note: Using resshift_nvidia.sif as it contains the evaluation environment
export CONTAINER="$VSC_SCRATCH/containers/resshift_nvidia.sif"
export EVAL_REPO="$VSC_DATA/evaluate"

# BCI specific paths
export PRED_DIR="/scratch/antwerpen/grp/ap_invilab_td_thesis/diffusion-predictions/ResShift/BCI"
export DATA_SQSH="$VSC_SCRATCH/datasets/BCI_dataset.sqsh"
export DATA_DIR="/tmp/BCI_eval_nvidia"
export OUTPUT_CSV="$VSC_DATA/projects/resshift/outputs/eval_results.csv"

module purge
module load calcua/2025a

echo "Starting BCI Evaluation on NVIDIA..."
mkdir -p "$DATA_DIR"

srun apptainer exec --nv \
    -B "$DATA_SQSH:$DATA_DIR:image-src=/" \
    -B "$VSC_DATA:$VSC_DATA" \
    -B "/scratch/antwerpen/grp/ap_invilab_td_thesis:/scratch/antwerpen/grp/ap_invilab_td_thesis" \
    "$CONTAINER" \
    bash -c "cd $EVAL_REPO && python evaluate.py \
        --pred $PRED_DIR \
        --gt $DATA_DIR/IHC/test \
        --model_name resshift_bci \
        --dataset_name bci \
        --split_name test \
        --output $OUTPUT_CSV"

echo "BCI Evaluation Complete."
