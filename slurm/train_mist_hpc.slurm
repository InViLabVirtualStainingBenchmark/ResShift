#!/bin/bash
#SBATCH --job-name=resshift_mist
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21213/projects/resshift/logs/train_mist_%x.%j.out
#SBATCH -e /data/antwerpen/212/vsc21213/projects/resshift/logs/train_mist_%x.%j.err

set -euo pipefail

# Pass stain at submit time: sbatch --export=ALL,STAIN=ER slurm/train_mist_hpc.slurm
if [ -z "${STAIN:-}" ]; then
    echo "ERROR: STAIN not set. Submit with: sbatch --export=ALL,STAIN=ER slurm/train_mist_hpc.slurm"
    exit 1
fi

STAIN_LOWER=$(echo "$STAIN" | tr '[:upper:]' '[:lower:]')

export PROJECT_NAME="ResShift"
export VSC_REPO="$VSC_DATA/projects/resshift/code/$PROJECT_NAME"
export CONTAINER="$VSC_SCRATCH/containers/resshift_hpc.sif"
export DATA_SQSH="$VSC_SCRATCH/datasets/MIST_dataset.sqsh"
export DATA_DIR="$VSC_SCRATCH/datasets/MIST"
export SAVE_DIR="$VSC_DATA/projects/resshift/outputs/experiments/mist_${STAIN_LOWER}"

mkdir -p "$DATA_DIR"
mkdir -p "$SAVE_DIR"

module purge
module load calcua/2025a

echo "Starting ResShift MIST Training — Stain: $STAIN"
echo "Job ID: $SLURM_JOB_ID"

if [ -n "${RESUME_CKPT:-}" ]; then
    echo "Resuming from : $RESUME_CKPT"
    RESUME_FLAG="--resume $RESUME_CKPT"
else
    echo "Starting fresh training."
    RESUME_FLAG=""
fi

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total \
           --format=csv -l 5 > "$VSC_DATA/projects/resshift/logs/gpu_mist_${STAIN_LOWER}_${SLURM_JOB_ID}.csv" &
GPU_LOG_PID=$!

srun apptainer exec --nv \
    -B "$DATA_SQSH:$DATA_DIR:image-src=/" \
    -B "$VSC_DATA:$VSC_DATA" \
    "$CONTAINER" \
    bash -c "cd $VSC_REPO && python main.py \
        --cfg_path configs/staining_mist_${STAIN_LOWER}.yaml \
        --save_dir $SAVE_DIR \
        $RESUME_FLAG"

kill $GPU_LOG_PID 2>/dev/null || true

echo ""
echo "$STAIN training complete."
echo "GPU log : $VSC_DATA/projects/resshift/logs/gpu_mist_${STAIN_LOWER}_${SLURM_JOB_ID}.csv"
echo "Results : $SAVE_DIR"
