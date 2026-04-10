import torch
import numpy as np
import random
import os
import pickle


import argparse

from config import DATA_DIR, N_CLASSES, PKL_FILE_DIR
from dataset import load_data
from models import ModelCtoy, ModelXtoC, ModelXtoCtoY, ModelXtoChat_ChatToY
from utils import AverageMeter
from utils_intervention import ATTRIBUTE_PARTS, ATTRIBUTES_IDX_USED, compute_concept_percentiles, intervene_on_attributes_random_trials
from utils_models import End2EndModel
from utils import log_and_store



if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")



K = [1, 3, 5] #top k class accuracies to compute



def run(args, log_lines):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
        
    if args.model_dir:
        print(f"Loading model from {args.model_dir}")
        # Need to change this if we want to load other types of models, but for now we only have the end2end model so this is fine
        if args.model_type == 'ModelXtoCtoY':
            model = ModelXtoCtoY(
                n_class_attr=args.n_class_attr,
                pretrained=False,
                num_classes=N_CLASSES,
                n_attributes=args.n_attributes,
                use_relu=args.use_relu,
                use_sigmoid=args.use_sigmoid
            )
        elif args.model_type == 'ModelCtoy':
            model = ModelCtoy(
                pretrained=False,
                freeze=False,
                input_dim=args.n_attributes,
                output_dim=N_CLASSES,
                expand_dim=args.expand_dim
            )
        elif args.model_type == 'ModelXtoC':
            model = ModelXtoC(
                pretrained=False,
                output_dim=args.n_attributes
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
    
    
    # When do we use model_dir2?
    if args.model_dir2:
        if args.model_type2 == 'ModelCtoy':
            model2 = ModelCtoy(
                pretrained=False,
                freeze=False,
                input_dim=args.n_attributes,
                output_dim=N_CLASSES,
                expand_dim=args.expand_dim
            )
        elif args.model_type2 == 'ModelXtoChat_ChatToY':
            model2 = ModelXtoChat_ChatToY(
                n_class_attr=args.n_class_attr,
                n_attributes=args.n_attributes,
                num_classes=N_CLASSES,
                expand_dim=args.expand_dim
                )
        

        state_dict2 = torch.load(args.model_dir2, map_location=device)
        model2 = model2.to(device)
        model2.load_state_dict(state_dict2)
        
        
        if not hasattr(model2, 'use_relu'):
            if args.use_relu:
                model2.use_relu = True
            else:
                model2.use_relu = False
        if not hasattr(model2, 'use_sigmoid'):
            if args.use_sigmoid:
                model2.use_sigmoid = True
            else:
                model2.use_sigmoid = False
        model2.eval()
    else:
        model2 = None
    


    # Need to change this
    eval_data_dir = args.data_dir + args.pkl_file_dir + args.eval_data + ".pkl"
    loader = load_data(args, [eval_data_dir], args.use_attr, args.no_img, args.batch_size, image_dir=args.image_dir,
                       n_class_attr=args.n_class_attr)
    
    ptl_5, ptl_95 = compute_concept_percentiles(args, model, loader)
    
    
    all_class_labels, all_attr_labels, all_attr_outputs = [], [], []
    all_attr_certainty = []
    
    selected_concepts_zero_based = torch.tensor([idx - 1 for idx in ATTRIBUTES_IDX_USED])
    test_data = pickle.load(open(args.data_dir + args.pkl_file_dir + "test" + ".pkl", "rb"))
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
            if isinstance(model, End2EndModel):
                attr_outputs = model.first_model(inputs)
            else:
                attr_outputs = model(inputs)

        all_class_labels.append(labels)
        all_attr_labels.append(attr_labels)
        all_attr_outputs.append(attr_outputs)  


    all_class_labels = torch.cat(all_class_labels, dim=0)
    all_attr_labels = torch.cat(all_attr_labels, dim=0)
    all_attr_outputs = torch.cat(all_attr_outputs, dim=0)
    
    print(f"all_class_labels shape: {all_class_labels.shape}, all_attr_labels shape: {all_attr_labels.shape}, all_attr_outputs shape: {all_attr_outputs.shape}, all_attr_certainty shape: {all_attr_certainty.shape}")
    
    
    
    
    
    
    
    accuracy_num_groups_intervened = []
    if args.selected_number_groups_intervene is not None:
        if not model2:
            accuracy = intervene_on_attributes_random_trials(
                model, args, all_attr_outputs, all_attr_labels, all_class_labels, ptl_5, ptl_95,args.selected_number_groups_intervene, all_attr_certainty, num_trials=args.num_trials)
        
        else:
            accuracy = intervene_on_attributes_random_trials(
                model2, args, all_attr_outputs, all_attr_labels, all_class_labels, ptl_5, ptl_95,args.selected_number_groups_intervene, all_attr_certainty, num_trials=args.num_trials)
        
        accuracy_num_groups_intervened.append(accuracy)
        
        print(f"Accuracy after intervening on {args.selected_number_groups_intervene} groups: {accuracy_num_groups_intervened[-1]}")
        log_lines.append(f"Accuracy after intervening on {args.selected_number_groups_intervene} groups: {accuracy_num_groups_intervened[-1]}")
    else:    
        for num_groups_intervene in range(len(ATTRIBUTE_PARTS) + 1):
            if not model2:
                accuracy = intervene_on_attributes_random_trials(
                    model, args, all_attr_outputs, all_attr_labels, all_class_labels, ptl_5, ptl_95,num_groups_intervene, all_attr_certainty, num_trials=args.num_trials)
            
            else:
                accuracy = intervene_on_attributes_random_trials(
                    model2, args, all_attr_outputs, all_attr_labels, all_class_labels, ptl_5, ptl_95,num_groups_intervene, all_attr_certainty, num_trials=args.num_trials)
            
            accuracy_num_groups_intervened.append(accuracy)
            
            print(f"Accuracy after intervening on {num_groups_intervene} groups: {accuracy_num_groups_intervened[-1]}")
            log_lines.append(f"Accuracy after intervening on {num_groups_intervene} groups: {accuracy_num_groups_intervened[-1]}")
            
        
        
    
if __name__ == '__main__':
    torch.backends.cudnn.benchmark=True

    parser = argparse.ArgumentParser(description='PyTorch Training')

    parser.add_argument('-output_file', default='results.txt', help='file name to save results in log_dir')
    parser.add_argument('-use_invisible', help='Whether to force intervention targets to 0 for attributes that are "not visible". Only applicable if attribute certainty labels are available.', action='store_true')
    parser.add_argument('-model_type', default='ModelXtoCtoY', help='which model architecture to use for intervention')
    parser.add_argument('-model_type2', default='ModelCtoy', help='which model architecture to use for intervention when we want to intervene on the output of the first model and feed it into a second model to predict class labels (e.g. for bottleneck model)')
    parser.add_argument('-expand_dim', default=0, type=int, help='the dimensionality of the hidden layer in the MLP. If 0, then no hidden layer and just a linear model. Only applicable for ModelCtoy architecture.')  
    parser.add_argument('-cub_data_dir', default='CUB_200_2011/', help='directory to the CUB image data')
    parser.add_argument('-pkl_file_dir', default='class_attr_data_10/', help='directory to the CUB pkl files relative to data_dir')
    parser.add_argument('-selected_number_groups_intervene', default=None, type=int, help='number of attribute groups to intervene on. If None, then will run intervention on all possible numbers of groups (from 0 to total number of groups)')
    parser.add_argument('-num_trials', default=5, type=int, help='number of random trials to run for each number of groups to intervene on (for random selection of groups to intervene on)')

    parser.add_argument('-log_dir', default='intervention', help='where results are stored')
    parser.add_argument('-model_dirs', default=None, nargs='+', help='where the trained models are saved')
    parser.add_argument('-model_dirs2', default=None, nargs='+', help='where another trained model are saved (for bottleneck only)')
    parser.add_argument('-eval_data', default='test', help='Type of data (train/ val/ test) to be used')
    parser.add_argument('-use_attr', help='whether tremove_attribute_indiceso use attributes (FOR COTRAINING ARCHITECTURE ONLY)', action='store_true')
    parser.add_argument('-no_img', help='if included, only use attributes (and not raw imgs) for class prediction', action='store_true')
    parser.add_argument('-bottleneck', help='whether to predict attributes before class labels', action='store_true')
    parser.add_argument('-image_dir', default='images', help='test image folder to run inference on')
    parser.add_argument('-n_class_attr', type=int, default=2, help='whether attr prediction is a binary or triary classification')
    parser.add_argument('-data_dir', default='Data/', help='directory to the data used for evaluation')
    parser.add_argument('-n_attributes', type=int, default=112, help='whether to apply bottlenecks to only a few attributes')    
    parser.add_argument('-attribute_group', default=None, help='file listing the (trained) model directory for each attribute group')
    parser.add_argument('-feature_group_results', help='whether to print out performance of individual atttributes', action='store_true')
    parser.add_argument('-use_relu', help='Whether to include relu activation before using attributes to predict Y. For end2end & bottleneck model', action='store_true')
    parser.add_argument('-use_sigmoid', help='Whether to include sigmoid activation before using attributes to predict Y. For end2end & bottleneck model', action='store_true')
    args = parser.parse_args()
    args.batch_size = 16
    
    args.model_dir = args.model_dirs[0]
    args.model_dir2 = args.model_dirs2[0] if args.model_dirs2 is not None else None

    # update args.n_attributes based on the data (in case of incomplete concept data, n_attributes will be different from total number of attributes)
    train_data = pickle.load(open(args.data_dir + args.pkl_file_dir + 'train.pkl', 'rb'))
    args.n_attributes = len(train_data[0]['attribute_label'])
    
    log_lines = []
    log_and_store(args, log_lines)
    
    run(args, log_lines)
    
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    
    with open(os.path.join(args.log_dir, args.output_file), 'w') as output:
        output.write('\n'.join(log_lines) + '\n')

  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    