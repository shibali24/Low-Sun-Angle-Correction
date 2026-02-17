#!/bin/bash
#SBATCH --job-name=planet-unet
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# -------- EDIT THESE FOR YOUR CLUSTER --------
PROJECT_DIR="${PROJECT_DIR:-$HOME/planet-unet}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-planet-unet}"
# ---------------------------------------------

mkdir -p "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}"

if [ -f "${CONDA_SH}" ]; then
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
fi

echo "Host: $(hostname)"
echo "Start: $(date)"
nvidia-smi || true

# Runtime controls
export FORCE_TRAIN=1
export RESTORE=0
export TRAINING_ITERS=60
export EPOCHS=10
export TRAIN_TIMEOUT_SEC=43200
export PRED_TIMEOUT_SEC=7200
export CONFIDENCE_THRESHOLD=0.60

# Paths
export MODEL_DIR="./out_paths/out_path_20191008-adam_l4-f64-dp75_Sentinel"
export TRAIN_DATA_DIR="./Sentinel-dense-train-test-split-all/train"
export TEST_DATA_DIR="./Sentinel-dense-train-test-split-all/test"
export PRED_DATA_DIR="${TEST_DATA_DIR}"

python run_pipeline_hpc.py

echo "End: $(date)"
