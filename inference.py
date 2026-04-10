"""
Evaluate trained models on the official CUB test set
"""
import os
import sys
import torch

import argparse
import numpy as np
from sklearn.metrics import f1_score
import pickle

from models import ModelCtoy, ModelXtoC, ModelXtoCtoY, ModelXtoChat_ChatToY
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dataset import load_data
from config import DATA_DIR, N_CLASSES, PKL_FILE_DIR
from utils import AverageMeter, multiclass_metric, accuracy, binary_accuracy, log_and_store
from utils_intervention import compute_concept_percentiles, intervene_on_attributes

import torch




if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")



K = [1, 3, 5] #top k class accuracies to compute

def eval(args, log_lines):
    """
    Run inference using model (and model2 if bottleneck)
    Returns: (for notebook analysis)
    all_class_labels: flattened list of class labels for each image
    topk_class_outputs: array of top k class ids predicted for each image. Shape = size of test set * max(K)
    all_class_outputs: array of all logit outputs for class prediction, shape = N_TEST * N_CLASS
    all_attr_labels: flattened list of labels for each attribute for each image (length = N_ATTRIBUTES * N_TEST)
    all_attr_outputs: flatted list of attribute logits (after ReLU/ Sigmoid respectively) predicted for each attribute for each image (length = N_ATTRIBUTES * N_TEST)
    all_attr_outputs_sigmoid: flatted list of attribute logits predicted (after Sigmoid) for each attribute for each image (length = N_ATTRIBUTES * N_TEST)
    wrong_idx: image ids where the model got the wrong class prediction (to compare with other models)
    """
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

    # I added and not args.no_img 
    if args.use_attr and not args.no_img:
        attr_acc_meter = [AverageMeter()]
        if args.feature_group_results:  # compute acc for each feature individually in addition to the overall accuracy
            for _ in range(args.n_attributes):
                attr_acc_meter.append(AverageMeter())
    else:
        attr_acc_meter = None

    class_acc_meter = []
    
    # K is the list of top k accuracies we want to compute, so we need a separate meter for each k
    for j in range(len(K)):
        class_acc_meter.append(AverageMeter())

    # Need to change this
    eval_data_dir = args.data_dir + args.pkl_file_dir + args.eval_data + ".pkl"

    loader = load_data(args, [eval_data_dir], args.use_attr, args.no_img, args.batch_size, image_dir=args.image_dir,
                       n_class_attr=args.n_class_attr)

    
    
    """
    if args.intervention:
        #train_data_dir = args.data_dir + args.pkl_file_dir + "train.pkl"
        #train_loader = load_data([train_data_dir], args.use_attr, args.no_img, args.batch_size, image_dir=args.image_dir,
                     #  n_class_attr=args.n_class_attr)
        ptl_5, ptl_95 = compute_concept_percentiles(model, loader)
    """
    

    all_attr_labels, all_attr_outputs, all_attr_outputs_sigmoid, all_attr_outputs2 = [], [], [], []
    all_class_labels, all_class_outputs, all_class_logits = [], [], []
    topk_class_labels, topk_class_outputs = [], []

    
    # Load data
    for data_idx, data in enumerate(loader):
        if args.use_attr:
            if args.no_img:  # A -> Y
                # inputs is the attribute labels
                inputs, labels = data
                if isinstance(inputs, list):
                    inputs = torch.stack(inputs).t().float()
                inputs = inputs.float()
                # inputs = torch.flatten(inputs, start_dim=1).float()
            # Use raw input, predicted concepets for final prediction, but still evaluate attribute prediction performance
            else:
                inputs, labels, attr_labels = data
                attr_labels = torch.stack(attr_labels).t()  # N x 312
                attr_labels = attr_labels.to(device)
        else:  # simple finetune
            inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        # Model for each attribute group?
        if args.attribute_group:
            outputs = []
            f = open(args.attribute_group, 'r')
            for line in f:
                attr_model = torch.load(line.strip())
                outputs.extend(attr_model(inputs))
                
        # Get attribute predictions and class predictions
        else:
            """
            if args.intervention:
                attr_outputs = model.first_model(inputs)
                #print(f"Intervening on attribute parts: {args.attribute_part_intervene}")
                attr_outputs_after_intervention = intervene_on_attributes(args, attr_outputs, attr_labels, ptl_5, ptl_95, args.attribute_part_intervene)
                class_outputs, _ = model.forward_stage2(attr_outputs_after_intervention)
                outputs = (class_outputs, attr_outputs) 
            """
            #else: 
            outputs = model(inputs)
        if args.use_attr:
            if args.no_img:  # A -> Y
                class_outputs = outputs
            else:
                if args.bottleneck:
                    if not args.model_type == "ModelXtoC":
                        attr_outputs = outputs[1:][0] # To get tensor instead of tuple of length 1 with tensor, Batch x N_ATTR
                    else: 
                        attr_outputs = outputs # for ModelXtoC, outputs is just the attribute predictions and there is no class prediction
                    if args.use_relu:
                        attr_outputs = torch.relu(attr_outputs)
                        attr_outputs_sigmoid = torch.sigmoid(attr_outputs)
                    elif args.use_sigmoid:
                        attr_outputs = torch.sigmoid(attr_outputs)
                        attr_outputs_sigmoid = attr_outputs
                    else:
                        attr_outputs_sigmoid = torch.sigmoid(attr_outputs)

                    if model2:
                        # stage2_inputs = torch.cat(attr_outputs, dim=1) # Do not think I need this I changed it 
                        #stage2_inputs = (attr_outputs_sigmoid >= 0.5).float() # need to threshold before concatenating for triary classification
                        stage2_inputs = attr_outputs
                        class_outputs = model2(stage2_inputs)
                    else:  # for debugging bottleneck performance without running stage 2
                        class_outputs = torch.zeros([inputs.size(0), N_CLASSES],
                                                    dtype=torch.float32).to(device)  # ignore this
                else:  # cotraining, end2end

                    attr_outputs = outputs[1:][0] # To get tensor instead of tuple of length 1 with tensor, Batch x N_ATTR
                    if args.use_relu:
                        attr_outputs = torch.relu(attr_outputs)
                        attr_outputs_sigmoid = torch.sigmoid(attr_outputs)
                    elif args.use_sigmoid:
                        attr_outputs = torch.sigmoid(attr_outputs)
                        attr_outputs_sigmoid = attr_outputs
                    else:
                        attr_outputs_sigmoid = torch.sigmoid(attr_outputs)

                    class_outputs = outputs[0]
                    
                    
                

                for i in range(args.n_attributes):
                    acc = binary_accuracy(attr_outputs_sigmoid[:, i].squeeze(), attr_labels[:, i])
                    acc = acc.data.cpu().numpy()
                    # acc = accuracy(attr_outputs_sigmoid[i], attr_labels[:, i], topk=(1,))
                    attr_acc_meter[0].update(acc, inputs.size(0))
                    if args.feature_group_results:  # keep track of accuracy of individual attributes
                        attr_acc_meter[i + 1].update(acc, inputs.size(0))

                all_attr_outputs.extend(list(attr_outputs.flatten().data.cpu().numpy()))
                all_attr_outputs_sigmoid.extend(list(attr_outputs_sigmoid.flatten().data.cpu().numpy()))
                all_attr_labels.extend(list(attr_labels.flatten().data.cpu().numpy()))
        else:
            class_outputs = outputs[0]

        # Top K predicted class ids
        _, topk_preds = class_outputs.topk(max(K), 1, True, True)
        # Top class prediction
        _, preds = class_outputs.topk(1, 1, True, True)
        all_class_outputs.extend(list(preds.detach().cpu().numpy().flatten()))
        all_class_labels.extend(list(labels.data.cpu().numpy()))
        all_class_logits.extend(class_outputs.detach().cpu().numpy())
        topk_class_outputs.extend(topk_preds.detach().cpu().numpy())
        topk_class_labels.extend(labels.view(-1, 1).expand_as(preds).cpu().numpy())

        np.set_printoptions(threshold=sys.maxsize)
        class_acc = accuracy(class_outputs, labels, topk=K)  # only class prediction accuracy
        for m in range(len(class_acc_meter)):
            class_acc_meter[m].update(class_acc[m], inputs.size(0))






    all_class_logits = np.vstack(all_class_logits)
    topk_class_outputs = np.vstack(topk_class_outputs)
    topk_class_labels = np.vstack(topk_class_labels)
    wrong_idx = np.where(np.sum(topk_class_outputs == topk_class_labels, axis=1) == 0)[0]

    for j in range(len(K)):
        log_and_store(f'Average top {K[j]} class accuracy: {class_acc_meter[j].avg.item():.5f}', log_lines)

    # Attribute prediction performance
    if args.use_attr and not args.no_img:  
        log_and_store(f'Average attribute accuracy: {attr_acc_meter[0].avg.item():.5f}', log_lines)
        all_attr_outputs_int = np.array(all_attr_outputs_sigmoid) >= 0.5
        if args.feature_group_results:
            n = len(all_attr_labels)
            all_attr_acc, all_attr_f1 = [], []
            for i in range(args.n_attributes):
                acc_meter = attr_acc_meter[1 + i]
                attr_acc = float(acc_meter.avg)
                attr_preds = [all_attr_outputs_int[j] for j in range(n) if j % args.n_attributes == i]
                attr_labels = [all_attr_labels[j] for j in range(n) if j % args.n_attributes == i]
                attr_f1 = f1_score(attr_labels, attr_preds)
                all_attr_acc.append(attr_acc)
                all_attr_f1.append(attr_f1)

            '''
            fig, axs = plt.subplots(1, 2, figsize=(20,10))
            for plt_id, values in enumerate([all_attr_acc, all_attr_f1]):
                axs[plt_id].set_xticks(np.arange(0, 1.1, 0.1))
                if plt_id == 0:
                    axs[plt_id].hist(np.array(values)/100.0, bins=np.arange(0, 1.1, 0.1), rwidth=0.8)
                    axs[plt_id].set_title("Attribute accuracies distribution")
                else:
                    axs[plt_id].hist(values, bins=np.arange(0, 1.1, 0.1), rwidth=0.8)
                    axs[plt_id].set_title("Attribute F1 scores distribution")
            plt.savefig('/'.join(args.model_dir.split('/')[:-1]) + '.png')
            '''
            bins = np.arange(0, 1.01, 0.1)
            acc_bin_ids = np.digitize(np.array(all_attr_acc) / 100.0, bins)
            acc_counts_per_bin = [np.sum(acc_bin_ids == (i + 1)) for i in range(len(bins))]
            f1_bin_ids = np.digitize(np.array(all_attr_f1), bins)
            f1_counts_per_bin = [np.sum(f1_bin_ids == (i + 1)) for i in range(len(bins))]
            print("Accuracy bins:")
            print(acc_counts_per_bin)
            print("F1 bins:")
            print(f1_counts_per_bin)
            np.savetxt(os.path.join(args.log_dir, 'concepts.txt'), f1_counts_per_bin)

        balanced_acc, report = multiclass_metric(all_attr_outputs_int, all_attr_labels)
        f1 = f1_score(all_attr_labels, all_attr_outputs_int)
        log_and_store(f"Total 1's predicted: {sum(np.array(all_attr_outputs_sigmoid) >= 0.5) / len(all_attr_outputs_sigmoid)}", log_lines)
        log_and_store(f"Avg attribute balanced acc: {balanced_acc}", log_lines)
        log_and_store(f"Avg attribute F1 score: {f1}", log_lines)
        log_and_store('Attribute Performance: \n' + report + '\n', log_lines)
    return class_acc_meter, attr_acc_meter, all_class_labels, topk_class_outputs, all_class_logits, all_attr_labels, all_attr_outputs, all_attr_outputs_sigmoid, wrong_idx, all_attr_outputs2

if __name__ == '__main__':
    torch.backends.cudnn.benchmark=True
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-intervention', help='Whether to intervene on attributes for a random subset of the test set, and compare performance on intervened vs non-intervened samples', action='store_true')
    parser.add_argument(
    "-attribute_part_intervene",
    nargs="+",          # one or more values
    type=str,           
    default=[],
    help="List of attribute parts to intervene on"
    )
    parser.add_argument('-output_file', default='results.txt', help='file name to save results in log_dir')
    parser.add_argument('-model_type', default='ModelXtoCtoY', help='type of model to evaluate, needed to determine how to load the model and what results to return. Only relevant if loading from checkpoint, otherwise can be ignored')
    parser.add_argument('-expand_dim', type=int, default=0,
                            help='dimension of hidden layer (if we want to increase model capacity) - for bottleneck only')
    parser.add_argument('-model_type2', default=None, help='type of model for second model to evaluate (for bottleneck), needed to determine how to load the model and what results to return.')
    parser.add_argument('-pkl_file_dir', default='class_attr_data_10/', help='directory to the CUB pkl files relative to data_dir')
    parser.add_argument('-cub_data_dir', default='CUB_200_2011/', help='directory to the CUB image data')
    
    
    
    parser.add_argument('-log_dir', default='.', help='where results are stored')
    parser.add_argument('-model_dirs', default=None, nargs='+', help='where the trained models are saved')
    parser.add_argument('-model_dirs2', default=None, nargs='+', help='where another trained model are saved (for bottleneck only)')
    parser.add_argument('-eval_data', default='test', help='Type of data (train/ val/ test) to be used')
    parser.add_argument('-use_attr', help='whether to use attributes (FOR COTRAINING ARCHITECTURE ONLY)', action='store_true')
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

    log_lines = []
    log_and_store(args, log_lines)
    y_results, c_results = [], []
    
    # update args.n_attributes based on the data (in case of incomplete concept data, n_attributes will be different from total number of attributes)
    train_data = pickle.load(open(args.data_dir + args.pkl_file_dir + 'train.pkl', 'rb'))
    args.n_attributes = len(train_data[0]['attribute_label'])
    
    
    
    for i, model_dir in enumerate(args.model_dirs):
        args.model_dir = model_dir
        args.model_dir2 = args.model_dirs2[i] if args.model_dirs2 else None
        result = eval(args, log_lines)
        class_acc_meter, attr_acc_meter = result[0], result[1]
        y_results.append(1 - class_acc_meter[0].avg[0].item() / 100.)
        if attr_acc_meter is not None:
            c_results.append(1 - attr_acc_meter[0].avg.item() / 100.)
        else:
            c_results.append(-1)
    values = (np.mean(y_results), np.std(y_results), np.mean(c_results), np.std(c_results))
    output_string = '%.4f %.4f %.4f %.4f' % values
    print_string = 'Error of y: %.4f +- %.4f, Error of C: %.4f +- %.4f' % values
    log_and_store(print_string, log_lines)
    log_and_store(output_string, log_lines)
    
    output_path = os.path.join(args.log_dir, args.output_file + ".txt") 
    with open(output_path, 'w') as output:
        output.write('\n'.join(log_lines) + '\n')

