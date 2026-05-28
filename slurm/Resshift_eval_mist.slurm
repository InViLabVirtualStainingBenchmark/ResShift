#!/bin/bash
#SBATCH --job-name=eval_mist_resshift
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=4:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p arcturus_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21213/projects/resshift/logs/eval_mist_%x.%j.out
#SBATCH -e /data/antwerpen/212/vsc21213/projects/resshift/logs/eval_mist_%x.%j.err

set -euo pipefail

# Submit with: sbatch --export=ALL,STAIN=ER slurm/eval_mist.slurm
if [ -z "${STAIN:-}" ]; then
    echo "ERROR: STAIN not set."
    exit 1
fi

STAIN_LOWER=$(echo "$STAIN" | tr '[:upper:]' '[:lower:]')

# Handle case-sensitivity for the Ki67 dataset folder inside the squashfs
if [ "$STAIN" == "KI67" ]; then
    STAIN_DIR="Ki67"
else
    STAIN_DIR="$STAIN"
fi

# Determine appropriate cellpose model based on stain biology
if [ "$STAIN" == "ER" ] || [ "$STAIN" == "PR" ] || [ "$STAIN" == "KI67" ]; then
    CP_MODEL="nuclei"
else
    CP_MODEL="cpsam" # Fallback or HER2
fi

# Paths
export CONTAINER="$VSC_SCRATCH/containers/evaluate_amd.sif"
export EVAL_REPO="$VSC_DATA/evaluate"

# MIST specific paths
export PRED_DIR="/scratch/antwerpen/grp/ap_invilab_td_thesis/diffusion-predictions/ResShift/MIST_${STAIN}"
export DATA_SQSH="$VSC_SCRATCH/datasets/MIST_dataset.sqsh"
export DATA_DIR="/tmp/MIST_eval"
export OUTPUT_CSV="$VSC_DATA/projects/resshift/outputs/eval_results.csv"

module purge
module load calcua/2025a

echo "Starting MIST ($STAIN) Evaluation on AMD..."
mkdir -p "$DATA_DIR"

srun apptainer exec --rocm \
    -B "$DATA_SQSH:$DATA_DIR:image-src=/" \
    -B "$VSC_DATA:$VSC_DATA" \
    -B "/scratch/antwerpen/grp/ap_invilab_td_thesis:/scratch/antwerpen/grp/ap_invilab_td_thesis" \
    "$CONTAINER" \
    bash -c "cd $EVAL_REPO && python evaluate.py \
        --pred $PRED_DIR \
        --gt $DATA_DIR/$STAIN_DIR/TrainValAB/valB \
        --model_name resshift_mist_${STAIN_LOWER} \
        --dataset_name mist \
        --split_name test \
        --cellpose \
        --cellpose_model $CP_MODEL \
        --output $OUTPUT_CSV"

echo "MIST ($STAIN) Evaluation Complete."