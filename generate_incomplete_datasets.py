import torch
from utils_intervention import get_attribute_parts_to_indices, ATTRIBUTE_PARTS
import random
import argparse
import os
import pickle

ATTRIBUTE_LABEL_LENGTH = 112


def create_random_incomplete_dataset(args, num_attribute_groups_remove=1):
    attribute_parts_indices_map = get_attribute_parts_to_indices(args)
    remove_attribute_parts = random.sample(ATTRIBUTE_PARTS, num_attribute_groups_remove)
    remove_attribute_indices = []
    for part in remove_attribute_parts:
        remove_attribute_indices.extend(attribute_parts_indices_map[part])  
        
    print(f"Removing attribute groups: {remove_attribute_parts} with attribute indices: {remove_attribute_indices}")
    
    # Create new pkl folder
    largest_digit = 0
    for folder_name in os.listdir(args.data_dir):
        if "class_attr_data_10_incomplete_" in folder_name:
            last_digit = folder_name.split("_")[-1]
            if last_digit.isdigit():
                largest_digit = max(largest_digit, int(last_digit))
    
    
    # New map from old attribute idx to new attribute idx (after removing some attributes)
    old_to_new_attr_idx = {}
    new_idx = 0

    for old_idx in range(ATTRIBUTE_LABEL_LENGTH):
        if old_idx not in remove_attribute_indices:
            old_to_new_attr_idx[old_idx] = new_idx
            new_idx += 1

    

    new_folder_path = os.path.join(args.data_dir, args.output_dir)
    os.makedirs(new_folder_path, exist_ok=True)
    
    with open(new_folder_path + "/info.txt", "w") as f:
        f.write(f"Removed attribute groups: {remove_attribute_parts}\n")
        f.write(f"Removed attribute indices: {sorted(remove_attribute_indices)}\n")
        f.write(f"Number of attribute groups removed: {num_attribute_groups_remove}\n")
        f.write(f"Total number of attributes removed: {len(remove_attribute_indices)}\n")
        f.write(f"New attribute indices mapping: {old_to_new_attr_idx}\n")
    
    # Modify pkl files and save to new folder
    pkl_files = ["train.pkl", "val.pkl", "test.pkl"]
    for pkl_file in pkl_files:
        pkl_path = os.path.join(args.data_dir + args.pkl_file_dir, pkl_file)
        data = pickle.load(open(pkl_path, "rb"))
        
        for sample in data:
            sample["attribute_label"] = [
                v for i, v in enumerate(sample["attribute_label"]) if i not in remove_attribute_indices
            ]
            """""
            sample["attribute_certainty"] = [
                v for i, v in enumerate(sample["attribute_certainty"]) if i not in remove_attribute_indices
            ]
            """
        new_pkl_path = os.path.join(new_folder_path, pkl_file)
        with open(new_pkl_path, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved modified {pkl_file} to {new_pkl_path}")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark=True

    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-cub_data_dir', default='CUB_200_2011/', help='directory to the CUB pkl files')
    parser.add_argument('-data_dir', default = 'Data/', help='root directory to data')
    parser.add_argument('-random', action='store_true', help='whether to create random incomplete dataset by removing random attribute groups')
    parser.add_argument('-num_attribute_groups_remove', default=1, type=int, help='number of attribute groups to remove for creating random incomplete dataset (only applicable if -random flag is set)')
    parser.add_argument('-pkl_file_dir', default='class_attr_data_10/', help='directory to the CUB pkl files relative to data_dir')
    parser.add_argument('-seed', default=42, type=int, help='random seed for reproducibility')
    parser.add_argument('-output_dir', default='Incomplete_Data/', help='directory to save the new pkl files with incomplete attribute data')

    args = parser.parse_args()
    
    
    
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    


    create_random_incomplete_dataset(args, args.num_attribute_groups_remove)
    
    