from config import CUB_DATA_DIR, DATA_DIR, PKL_FILE_DIR, N_CLASSES, N_ATTRIBUTES
import pickle
import numpy as np
import random
import torch

from utils import accuracy
from utils_models import End2EndModel

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


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

"""
def get_attribute_class_statistics():
    train_data = pickle.load(open(args.data_dir + args.pkl_file_dir + "train_data.pkl", "rb"))
    
    # Count samples in each class with each attribute present/absent
    class_attr_count = np.zeros((N_CLASSES, N_ATTRIBUTES, 2)) 
    
    
    # Build statistics about attribute presence/absence in each class
    for sample in train_data:
        class_id = sample['class_id']
        attr_label = sample['attribute_label']
        attr_certainty = sample['attribute_certainty']
        for attr_idx, attr_active_status in enumerate(attr_label):
            # Attribute is marked as absent but we are not confident as attribute is not visible 
            if attr_active_status == 0 and attr_certainty[attr_idx] == 1:
                continue
            class_attr_count[class_id, attr_idx, int(attr_active_status)] += 1
            
    class_attr_min_label = np.argmin(class_attr_count, axis=2) # More absent observations than present for each class and attribute, returns the first index of the minimum value in case of tie
    class_attr_max_label = np.argmax(class_attr_count, axis=2) # More present observations than absent for each class and attribute, returns the first index of the maximum value in case of tie
    equal_count = np.where(class_attr_min_label == class_attr_max_label)  # check where 0 count = 1 count, set the corresponding class attribute label to be 1
    class_attr_max_label[equal_count] = 1 # In case of tie, set to 1 (present)

    return class_attr_max_label # (N_CLASSES, N_ATTRIBUTES)
"""


def get_attribute_mask(class_attr_labels, min_class_count=10):
    """
    Keep attributes that are present at the class level
    in at least `min_class_count` classes.
    """
    attr_class_count = np.sum(class_attr_labels, axis=0)
    mask = np.where(attr_class_count >= min_class_count)[0]
    return mask, attr_class_count


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


def intervene_on_attributes(args, attr_logits, attr_labels, ptl_5, ptl_95, attribute_part_intervene: list):
    """
    attr_outputs: tensor of shape (batch_size, n_attributes)
    intervention_dict: dict mapping attribute indices to intervention values (0 or 1)
    """
 

    if not attribute_part_intervene:
        
        raise KeyError("Need to specify attribute_part_intervene as a list of attribute parts to intervene on (e.g. ['has_bill_shape', 'has_wing_color'])")
    
    
    
    # Get attribute idx to intervene 
    attribute_parts_to_indices = get_attribute_parts_to_indices(args)
    intervene_idx = []
    
    for part_name in attribute_part_intervene:
        if part_name not in ATTRIBUTE_PARTS:
            raise KeyError(f"Invalid attribute part name: {part_name}.")
        part_attr_indices = attribute_parts_to_indices[part_name]
        intervene_idx.extend(part_attr_indices)

    B, A = attr_logits.shape # batch size, number of attributes
    attr_new = attr_logits.clone()
    
    #print(intervene_idx)
        
    for i in range(B):
        for a in intervene_idx:
            binary_val = attr_labels[i, a].item()
            if binary_val == 1:
                attr_new[i, a] = float(ptl_95[a])
            else:
                attr_new[i, a] = float(ptl_5[a])

    
        
    return attr_new


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

    
    
    for _ in range(num_trials):
        attr_new = attr_logits.clone()

        for i in range(B):
            # intervene random groups of attributes per image
            parts = random.sample(ATTRIBUTE_PARTS, n_groups_replace)

            intervene_idx = []
            for part_name in parts:
                intervene_idx.extend(attribute_parts_to_indices[part_name])

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







    
    
    
    
    
   

