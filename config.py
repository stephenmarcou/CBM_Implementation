import os
# Directory paths
CUB_DATA_DIR = "CUB_200_2011/"
PKL_FILE_DIR = "class_attr_data_10/"
PKL_FILE_INCOMPLETE_DIR = "class_attr_data_incomplete/"
ROOT_LOG_DIR = os.getenv("ROOT_LOG_DIR", "./Logs/")
DATA_DIR = os.getenv("DATA_DIR", "./Data/")



N_CLASSES = 200
N_ATTRIBUTES_ORIG = 312

# Training
MIN_LR = 0.0001
LR_DECAY_SIZE = 0.1