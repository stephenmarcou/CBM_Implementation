#!/bin/bash
#SBATCH --job-name=independent_c_to_y
#SBATCH --output=/cluster/home/smarcou/Desktop/Work\ \(biomed\)/vogtlab/Group/smarcou/logs/%x_%j.out
#SBATCH --error=/cluster/home/smarcou/Desktop/Work\ \(biomed\)/vogtlab/Group/smarcou/logs/%x_%j.err
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --time=00:10:00

set -euo pipefail

# -------------------------
# Paths
# -------------------------
BASE_DIR="/cluster/customapps/biomed/vogtlab/users/smarcou/CBM_Implementation"
LOG_DIR="/cluster/home/smarcou/Desktop/Work (biomed)/vogtlab/Group/smarcou/Logs"
CUB_DATA_DIR="/cluster/home/smarcou/Desktop/Work (biomed)/vogtlab/Group/smarcou/CUB_Data"

# -------------------------
# Runtime parameters
# -------------------------
SEED=${SEED:-1}
LR=${LR:-0.001}
EPOCHS=${EPOCHS:-30}
BATCH_SIZE=${BATCH_SIZE:-64}

export ROOT_LOG_DIR="$LOG_DIR"
export CUB_DATA_DIR="$CUB_DATA_DIR"

# -------------------------
# Go to project directory
# -------------------------
cd "$BASE_DIR" || exit 1

# -------------------------
# Activate environment
# -------------------------
source /cluster/customapps/biomed/vogtlab/users/smarcou/software/anaconda/bin/activate
conda activate CBM_implementation_venv

# -------------------------
# Debug info
# -------------------------
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"
echo "ROOT_LOG_DIR: $ROOT_LOG_DIR"
echo "CUB_DATA_DIR: $CUB_DATA_DIR"
echo "Python path: $(which python)"
echo "Python version: $(python --version)"

# -------------------------
# Run training
# -------------------------
echo "Starting training..."

python main.py cub Independent_CtoY \
    --seed "$SEED" \
    -log_dir c_to_y \
    -e "$EPOCHS" \
    -optimizer sgd \
    -use_attr \
    -n_attributes 112 \
    -no_img \
    -b "$BATCH_SIZE" \
    -weight_decay 0.00005 \
    -lr "$LR" \
    -scheduler_step 1000