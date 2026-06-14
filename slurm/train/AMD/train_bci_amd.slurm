#!/bin/bash
#SBATCH --job-name=train_bci_amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p arcturus_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21213/projects/resshift/logs/train_bci_amd.%j.out
#SBATCH -e /data/antwerpen/212/vsc21213/projects/resshift/logs/train_bci_amd.%j.err

set -euo pipefail

export PROJECT_NAME="ResShift"
export VSC_REPO="$VSC_DATA/projects/resshift/code/$PROJECT_NAME"
export CONTAINER="$VSC_SCRATCH/containers/resshift_amd.sif"
export DATA_SQSH="$VSC_SCRATCH/datasets/BCI_dataset.sqsh"
export DATA_DIR="$VSC_SCRATCH/datasets/BCI"
export SAVE_DIR="$VSC_DATA/projects/resshift/outputs/experiments/bci"

# MIOPEN cache redirection
export MIOPEN_USER_DB_PATH=/tmp/miopen_$SLURM_JOB_ID
export MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen_$SLURM_JOB_ID
mkdir -p $MIOPEN_USER_DB_PATH

mkdir -p "$SAVE_DIR"
mkdir -p "$DATA_DIR"

module purge
module load calcua/2025a

echo "Starting ResShift BCI Training (AMD) on arcturus_gpu..."

if [ -n "${RESUME_CKPT:-}" ]; then
    RESUME_FLAG="--resume $RESUME_CKPT"
else
    RESUME_FLAG=""
fi

while true; do
    python3 -c "
import subprocess, datetime
r = subprocess.run(['rocm-smi', '-u', '-m', '-P', '-t'], capture_output=True, text=True)
print(datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S'), '\n', r.stdout)
" >> "$VSC_DATA/projects/resshift/logs/gpu_bci_amd_${SLURM_JOB_ID}.log"
    sleep 5
done &
GPU_LOG_PID=$!

srun apptainer exec --rocm \
    -B "$DATA_SQSH:$DATA_DIR:image-src=/" \
    -B "$VSC_DATA:$VSC_DATA" \
    "$CONTAINER" \
    bash -c "cd $VSC_REPO && python main.py \
        --cfg_path configs/staining_bci.yaml \
        --save_dir $SAVE_DIR \
        $RESUME_FLAG"

kill $GPU_LOG_PID 2>/dev/null || true
rm -rf $MIOPEN_USER_DB_PATH

echo "BCI AMD training complete."
echo "GPU log : $VSC_DATA/projects/resshift/logs/gpu_bci_amd_${SLURM_JOB_ID}.log"
echo "Results : $SAVE_DIR"
