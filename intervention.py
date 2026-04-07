import torch
import numpy as np
import random
import os
import pickle


import argparse

from config import DATA_DIR, N_CLASSES, PKL_FILE_DIR
from dataset import load_data
from models import ModelXtoCtoY
from utils import AverageMeter
from utils_intervention import ATTRIBUTE_PARTS, ATTRIBUTES_IDX_USED, compute_concept_percentiles, intervene_on_attributes_random_trials


if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")



K = [1, 3, 5] #top k class accuracies to compute



def run(args):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
        
    if args.model_dir:
        # Need to change this if we want to load other types of models, but for now we only have the end2end model so this is fine
        model = ModelXtoCtoY(
            n_class_attr=args.n_class_attr,
            pretrained=False,
            num_classes=N_CLASSES,
            n_attributes=args.n_attributes,
            use_relu=args.use_relu,
            use_sigmoid=args.use_sigmoid
        )
        state_dict = torch.load(args.model_dir, map_location=device)
        model = model.to(device)
        model.load_state_dict(state_dict)
    
    else:
        model = None

    if not hasattr(model, 'use_relu'):
        if args.use_relu:
            model.use_relu = True
        else:
            model.use_relu = False
    if not hasattr(model, 'use_sigmoid'):
        if args.use_sigmoid:
            model.use_sigmoid = True
        else:
            model.use_sigmoid = False
    if not hasattr(model, 'cy_fc'):
        model.cy_fc = None
    model.eval()
    


    # Need to change this
    eval_data_dir = DATA_DIR + PKL_FILE_DIR + args.eval_data + ".pkl"
    loader = load_data([eval_data_dir], args.use_attr, args.no_img, args.batch_size, image_dir=args.image_dir,
                       n_class_attr=args.n_class_attr)
    
    ptl_5, ptl_95 = compute_concept_percentiles(model, loader)
    
    
    all_class_labels, all_attr_labels, all_attr_outputs = [], [], []
    all_attr_certainty = []
    
    selected_concepts_zero_based = torch.tensor([idx - 1 for idx in ATTRIBUTES_IDX_USED])
    test_data = pickle.load(open(DATA_DIR + PKL_FILE_DIR + "test" + ".pkl", "rb"))
    for sample in test_data:
        attribute_certainty_kept = torch.tensor(sample['attribute_certainty'])[selected_concepts_zero_based]
        all_attr_certainty.append(attribute_certainty_kept)
        #print(type(attribute_certainty_kept), len(attribute_certainty_kept))

    all_attr_certainty = torch.stack(all_attr_certainty)
    
    
    
    
    
    for i, data in enumerate(loader):
        inputs, labels, attr_labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        # Convert attr_labels to tensor if it's a list or numpy array
        attr_labels = torch.stack(attr_labels, dim=1)
        #print(f"attr_labels shape: {attr_labels.shape}, inputs shape: {inputs.shape}, labels shape:  {labels.shape}")

        attr_labels = attr_labels.to(device)

        with torch.no_grad():
            attr_outputs = model.first_model(inputs)

        all_class_labels.append(labels)
        all_attr_labels.append(attr_labels)
        all_attr_outputs.append(attr_outputs)  


    all_class_labels = torch.cat(all_class_labels, dim=0)
    all_attr_labels = torch.cat(all_attr_labels, dim=0)
    all_attr_outputs = torch.cat(all_attr_outputs, dim=0)
    
    print(f"all_class_labels shape: {all_class_labels.shape}, all_attr_labels shape: {all_attr_labels.shape}, all_attr_outputs shape: {all_attr_outputs.shape}, all_attr_certainty shape: {all_attr_certainty.shape}")
    
    
    
    
    
    
    
    accuracy_num_groups_intervened = []
    for num_groups_intervene in range(len(ATTRIBUTE_PARTS) + 1):
        accuracy = intervene_on_attributes_random_trials(
            model, args, all_attr_outputs, all_attr_labels, all_class_labels, ptl_5, ptl_95,num_groups_intervene, all_attr_certainty)
        
        accuracy_num_groups_intervened.append(accuracy)
        
        print(f"Accuracy after intervening on {num_groups_intervene} groups: {accuracy_num_groups_intervened[-1]}")
        
        
        
        
    
if __name__ == '__main__':
    torch.backends.cudnn.benchmark=True

    parser = argparse.ArgumentParser(description='PyTorch Training')

    parser.add_argument('-output_file', default='results.txt', help='file name to save results in log_dir')
    parser.add_argument('-use_invisible', help='Whether to force intervention targets to 0 for attributes that are "not visible". Only applicable if attribute certainty labels are available.', action='store_true')

    parser.add_argument('-log_dir', default='.', help='where results are stored')
    parser.add_argument('-model_dirs', default=None, nargs='+', help='where the trained models are saved')
    parser.add_argument('-model_dirs2', default=None, nargs='+', help='where another trained model are saved (for bottleneck only)')
    parser.add_argument('-eval_data', default='test', help='Type of data (train/ val/ test) to be used')
    parser.add_argument('-use_attr', help='whether to use attributes (FOR COTRAINING ARCHITECTURE ONLY)', action='store_true')
    parser.add_argument('-no_img', help='if included, only use attributes (and not raw imgs) for class prediction', action='store_true')
    parser.add_argument('-bottleneck', help='whether to predict attributes before class labels', action='store_true')
    parser.add_argument('-image_dir', default='images', help='test image folder to run inference on')
    parser.add_argument('-n_class_attr', type=int, default=2, help='whether attr prediction is a binary or triary classification')
    parser.add_argument('-data_dir', default='', help='directory to the data used for evaluation')
    parser.add_argument('-n_attributes', type=int, default=112, help='whether to apply bottlenecks to only a few attributes')    
    parser.add_argument('-attribute_group', default=None, help='file listing the (trained) model directory for each attribute group')
    parser.add_argument('-feature_group_results', help='whether to print out performance of individual atttributes', action='store_true')
    parser.add_argument('-use_relu', help='Whether to include relu activation before using attributes to predict Y. For end2end & bottleneck model', action='store_true')
    parser.add_argument('-use_sigmoid', help='Whether to include sigmoid activation before using attributes to predict Y. For end2end & bottleneck model', action='store_true')
    args = parser.parse_args()
    args.batch_size = 16
    
    args.model_dir = args.model_dirs[0]
    args.model_dir2 = args.model_dirs2[0] if args.model_dirs2 is not None else None

    print(args)
    
    run(args)

  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
def simulate_group_intervention_random(
    replace_val,
    ptl_5,
    ptl_95,
    model2,
    attr_group_dict,
    b_class_labels,
    b_class_logits,
    b_attr_outputs,
    b_attr_labels,
    instance_attr_labels,
    uncertainty_attr_labels,
    use_not_visible,
    n_replace,
    use_relu,
    use_sigmoid,
    n_trials=1,
    connect_CY=False,
):
    assert len(instance_attr_labels) == len(b_attr_labels), (
        'len(instance_attr_labels): %d, len(b_attr_labels): %d'
        % (len(instance_attr_labels), len(b_attr_labels))
    )
    assert len(uncertainty_attr_labels) == len(b_attr_labels), (
        'len(uncertainty_attr_labels): %d, len(b_attr_labels): %d'
        % (len(uncertainty_attr_labels), len(b_attr_labels))
    )

    all_class_acc = []

    for _ in range(n_trials):
        b_attr_new = np.array(b_attr_outputs[:])

        def replace_random():
            replace_idx = []
            group_replace_idx = list(random.sample(list(range(args.n_groups)), n_replace))
            for group_id in group_replace_idx:
                replace_idx.extend(attr_group_dict[group_id])
            return replace_idx

        attr_replace_idx = []

        for img_id in range(len(b_class_labels)):
            replace_idx = replace_random()
            attr_replace_idx.extend(np.array(replace_idx) + img_id * args.n_attributes)

        if replace_val == 'class_level':
            b_attr_new[attr_replace_idx] = np.array(b_attr_labels)[attr_replace_idx]
        else:
            b_attr_new[attr_replace_idx] = np.array(instance_attr_labels)[attr_replace_idx]

        if use_not_visible:
            not_visible_idx = np.where(np.array(uncertainty_attr_labels) == 1)[0]
            for idx in attr_replace_idx:
                if idx in not_visible_idx:
                    b_attr_new[idx] = 0

        if use_relu or not use_sigmoid:
            binary_vals = b_attr_new[attr_replace_idx]
            for j, replace_idx in enumerate(attr_replace_idx):
                attr_idx = replace_idx % args.n_attributes
                b_attr_new[replace_idx] = (
                    (1 - binary_vals[j]) * ptl_5[attr_idx]
                    + binary_vals[j] * ptl_95[attr_idx]
                )

        model2.eval()

        b_attr_new = b_attr_new.reshape(-1, args.n_attributes)
        stage2_inputs = torch.from_numpy(np.array(b_attr_new)).to(device)

        if connect_CY:
            new_cy_outputs = model2(stage2_inputs)
            old_stage2_inputs = torch.from_numpy(
                np.array(b_attr_outputs).reshape(-1, args.n_attributes)
            ).to(device)
            old_cy_outputs = model2(old_stage2_inputs)
            class_outputs = torch.from_numpy(b_class_logits).to(device) + (new_cy_outputs - old_cy_outputs)
        else:
            class_outputs = model2(stage2_inputs)

        _, preds = class_outputs.topk(1, 1, True, True)
        b_class_outputs_new = preds.data.cpu().numpy().squeeze()
        class_acc = np.mean(np.array(b_class_outputs_new) == np.array(b_class_labels))
        all_class_acc.append(class_acc * 100)

    return max(all_class_acc)