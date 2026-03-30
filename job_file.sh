#SBATCH --job-name=test_cluster
#SBATCH --output="/cluster/home/smarcou/Desktop/Work (biomed)/vogtlab/Group/smarcou/logs/test_cluster_%j.out"
#SBATCH --error="/cluster/home/smarcou/Desktop/Work (biomed)/vogtlab/Group/smarcou/logs/test_cluster_%j.err"    
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=2G
#SBATCH --time=00:10:00

# Activate virtual env
source "/cluster/home/smarcou/Desktop/Work (biomed)/vogtlab/Group/smarcou/envs/test_cluster_venv/bin/activate"

# -------------------------
# Export environment variables
# -------------------------
export ROOT_LOG_DIR="/cluster/home/smarcou/Desktop/Work (biomed)/vogtlab/Group/smarcou/Logs/"
export CUB_DATA_DIR="/cluster/home/smarcou/Desktop/Work (biomed)/vogtlab/Group/smarcou/CUB_Data/"


# -------------------------
# Activate virtual environment
# -------------------------
source /cluster/customapps/biomed/vogtlab/users/smarcou/software/anaconda/bin/activate
conda activate CBM_implementation_venv
pip install -r requirements.txt


# -------------------------
# Debug info
# -------------------------
echo "Running on host: $(hostname)"
echo "Current working directory: $(pwd)"
echo "ROOT_LOG_DIR: $ROOT_LOG_DIR"
echo "DATA_DIR: $CUB_DATA_DIR"
echo "Python version: $(python --version)"

# -------------------------
# Run your training command
# -------------------------
echo "Starting training..."

python3 main.py cub Independent_CtoY \
    --seed 1 \
    -log_dir c_to_y \
    -e 30 \
    -optimizer sgd \
    -use_attr \
    -n_attributes 112 \
    -no_img \
    -b 64 \
    -weight_decay 0.00005 \
    -lr 0.001 \
    -scheduler_step 1000