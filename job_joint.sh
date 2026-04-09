#!/bin/bash
#SBATCH --job-name=joint_attr_w001
#SBATCH --output="/cluster/home/smarcou/Desktop/Work (biomed)/vogtlab/Group/smarcou/Logs/%x_%j.out"
#SBATCH --error="/cluster/home/smarcou/Desktop/Work (biomed)/vogtlab/Group/smarcou/Logs/%x_%j.err"
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=08:00:00

set -euo pipefail

# -------------------------
# Paths
# -------------------------
BASE_DIR="/cluster/customapps/biomed/vogtlab/users/smarcou/CBM_Implementation/"
LOG_DIR="/cluster/home/smarcou/Desktop/Work (biomed)/vogtlab/Group/smarcou/Logs/"
CUB_DATA_DIR="/cluster/home/smarcou/Desktop/Work (biomed)/vogtlab/Group/smarcou/CUB_Data/"

# -------------------------
# Runtime parameters (override at submit time if needed)
# Example: sbatch --export=ALL,SEED=2,LR=0.0005 job_joint.sh
# -------------------------
SEED=${SEED:-1}
LR=${LR:-0.001}
EPOCHS=${EPOCHS:-40}
BATCH_SIZE=${BATCH_SIZE:-64}
ATTR_LOSS_WEIGHT=${ATTR_LOSS_WEIGHT:-0.01}
LOG_DIR_CL=${LOG_DIR:-$BASE_DIR/Logs/joint_model_attr_weight_0.01}

export ROOT_LOG_DIR="$LOG_DIR"

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
echo "Python path: $(which python3)"
echo "Python version: $(python3 --version)"


# -------------------------
# Run training
# -------------------------
echo "Starting Joint training..."

python3 main.py cub Joint \
    --seed "$SEED" \
    -log_dir "$LOG_DIR_CL" \
    -e "$EPOCHS" \
    -optimizer sgd \
    -momentum 0.9 \
    -pretrained \
    -use_attr \
    -weighted_loss multiple \
    -n_attributes 112 \
    -attr_loss_weight "$ATTR_LOSS_WEIGHT" \
    -b "$BATCH_SIZE" \
    -weight_decay 0.00004 \
    -lr "$LR" \
    -scheduler_step 15 \
    -end2end \
    -print_attr_acc \
    -data_dir "$CUB_DATA_DIR"
