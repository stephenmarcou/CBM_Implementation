from config import CUB_DATA_DIR, DATA_DIR, PKL_FILE_DIR, N_CLASSES, N_ATTRIBUTES
import pickle
import numpy as np
import random
import torch
import re

from utils import accuracy
from utils_models import End2EndModel

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


# Fom CEM repo
ATTRIBUTE_PARTS = [
 'has_bill_shape',
 'has_wing_color',
 'has_upperparts_color',
 'has_underparts_color',
 'has_breast_pattern',
 'has_back_color',
 'has_tail_shape',
 'has_upper_tail_color',
 'has_head_pattern',
 'has_breast_color',
 'has_throat_color',
 'has_eye_color',
 'has_bill_length',
 'has_forehead_color',
 'has_under_tail_color',
 'has_nape_color',
 'has_belly_color',
 'has_wing_shape',
 'has_size',
 'has_shape',
 'has_back_pattern',
 'has_tail_pattern',
 'has_belly_pattern',
 'has_primary_color',
 'has_leg_color',
 'has_bill_color',
 'has_crown_color',
 'has_wing_pattern',
]

# Fom CEM repo
ATTRIBUTES_IDX_USED = [1, 4, 6, 7, 10, 14, 15, 20, 21, 23, 25, 29, 30, 35, 36, 38, 40, 44, 45, 50, 51, 53, 54, 56, 57, 59, 63, 64, 69, 70, 72, 75, 80, 84, 90, 91, \
    93, 99, 101, 106, 110, 111, 116, 117, 119, 125, 126, 131, 132, 134, 145, 149, 151, 152, 153, 157, 158, 163, 164, 168, 172, 178, 179, 181, \
    183, 187, 188, 193, 194, 196, 198, 202, 203, 208, 209, 211, 212, 213, 218, 220, 221, 225, 235, 236, 238, 239, 240, 242, 243, 244, 249, 253, \
    254, 259, 260, 262, 268, 274, 277, 283, 289, 292, 293, 294, 298, 299, 304, 305, 308, 309, 310, 311]



def compute_concept_percentiles(args, model, loader):
    model.eval()
    all_attr_outputs = []

    with torch.no_grad():
        for data in loader:
            inputs, labels, attr_labels = data
            inputs = inputs.to(device)

            if args.model_type == 'ModelXtoC':
                attr_outputs = model(inputs)
            else:
                attr_outputs = model.first_model(inputs)


            all_attr_outputs.append(attr_outputs.detach().cpu().numpy())

    all_attr_outputs = np.concatenate(all_attr_outputs, axis=0)  # (N, A)
    ptl_5 = np.percentile(all_attr_outputs, 5, axis=0)
    ptl_95 = np.percentile(all_attr_outputs, 95, axis=0)
    return ptl_5, ptl_95





def get_attribute_parts_to_indices(args):
    """
    Maps attribute idx to attribute parts (e.g. bill, wing, etc.)
    """
    
    
    # Attribute idx in the original 312 attribute space to attribute idx in the new 112 attribute space
    old_idx_to_new_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(ATTRIBUTES_IDX_USED)}
    #print(old_idx_to_new_idx)
    
    with open(args.data_dir + args.cub_data_dir + "attributes.txt", "r") as f:
        lines = f.readlines()
        semantic_groups = {}
        for line in lines:
            idx, attr_name = line.strip().split(" ")
            idx = int(idx) - 1 # Convert to 0-based index
            if idx not in ATTRIBUTES_IDX_USED:
                continue
                
            new_idx = old_idx_to_new_idx[idx]
            semantic_group = attr_name.split("::")[0]
            if semantic_group not in semantic_groups:
                semantic_groups[semantic_group] = []
            semantic_groups[semantic_group].append(new_idx)
        #print(semantic_groups)
        return semantic_groups





# For incomplete data, from info file find which attribute parts were kept
def get_kept_attribute_parts(args):
    from utils_intervention import ATTRIBUTE_PARTS
    info_file = args.data_dir + args.pkl_file_dir + "info.txt"
    with open(info_file, "r") as f:
        lines = f.readlines()
        line_attributes_removed = lines[0]
        attributes_removed = re.findall(r"'([^']*)'", line_attributes_removed)
        attributes_kept = [attr for attr in ATTRIBUTE_PARTS if attr not in attributes_removed]
        print(attributes_kept)
        print(len(attributes_kept))
        return attributes_kept
    
# For incomplete data, from info file find the mapping from old attribute idx to new attribute idx after removing some attributes
def get_map_from_old_to_new_attribute_idx(args):
    info_file = args.data_dir + args.pkl_file_dir + "info.txt"
    with open(info_file, "r") as f:
        lines = f.readlines()
        pairs = re.findall(r"(\d+):\s*(\d+)", lines[-1])
        old_idx_to_new_idx = {int(k): int(v) for k, v in pairs}
        return old_idx_to_new_idx



def intervene_on_attributes_random_trials(
    model,
    args,
    attr_logits,
    attr_labels,
    class_labels,
    ptl_5,
    ptl_95,
    n_groups_replace,
    attr_certainty,
    num_trials=1,
):
    attribute_parts_to_indices = get_attribute_parts_to_indices(args)
    accuracy_trials = []
    
    device = attr_logits.device
    ptl_5 = torch.tensor(ptl_5, device=device)
    ptl_95 = torch.tensor(ptl_95, device=device)

    B, A = attr_logits.shape
    
    if args.incomplete:
        attribute_parts = get_kept_attribute_parts(args)
    else:
        attribute_parts = ATTRIBUTE_PARTS

    if n_groups_replace > len(attribute_parts):
        return -1
    
    for _ in range(num_trials):
        attr_new = attr_logits.clone()

        for i in range(B):
            # intervene random groups of attributes per image
            parts = random.sample(attribute_parts, n_groups_replace)

            intervene_idx = []
            for part_name in parts:
                intervene_idx.extend(attribute_parts_to_indices[part_name])

            # If incomplete then have to update the idx of the intervene_idx to match the new attribute idx after removing some attributes
            if args.incomplete:
                old_idx_to_new_idx = get_map_from_old_to_new_attribute_idx(args)
                intervene_idx = [old_idx_to_new_idx[a] for a in intervene_idx]

            for a in intervene_idx:
                binary_val = attr_labels[i, a].item()
                # if attribute is "not visible", force intervention target to 0
                if args.use_invisible and attr_certainty is not None and attr_certainty[i, a] == 1:
                    binary_val = 0
                if binary_val == 1:
                    attr_new[i, a] = ptl_95[a]

                    
                else:
                    attr_new[i, a] = ptl_5[a]
            
        
        if isinstance(model, End2EndModel):
            print("Using forward_stage2 for End2EndModel")
            class_outputs, _ = model.forward_stage2(attr_new)
        else:
            class_outputs = model(attr_new)
        acc = accuracy(class_outputs, class_labels)
        acc = acc[0].item()  # Get the accuracy value from the list of tensors
        accuracy_trials.append(acc)
    print(accuracy_trials, type(max(accuracy_trials)))
    return max(accuracy_trials)







    
    
    
    
    
   

