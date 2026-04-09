import torch
import argparse
from models import ModelXtoC, ModelCtoy, ModelXtoChat_ChatToY
from dataset import load_data
import os
import pickle

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available(): 
    device = torch.device("mps")
else:
    device = torch.device("cpu")

def extract_concepts(args):
    model = ModelXtoC(pretrained=args.pretrained, output_dim=args.n_attributes)
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)
    model.eval()
    
    pkl_files = ["train.pkl", "val.pkl", "test.pkl"]
    
    for pkl_file in pkl_files:
        full_path = args.data_dir + args.pkl_file_dir + pkl_file
        
        loader = load_data(args, [full_path], args.use_attr, args.no_img, args.batch_size, create_new_dataset=True) 
        
        attr_logits = []
        
        with torch.no_grad():
            for data_idx, data in enumerate(loader):
                inputs, labels, attr_labels = data
                inputs = inputs.to(device)
                attr_outputs = model(inputs)
                if data_idx == 0:
                    print(attr_outputs)
                attr_logits.append(attr_outputs.detach().cpu())
        attr_logits = torch.cat(attr_logits, dim=0).numpy()  # (N, A)
        
        orig_data = pickle.load(open(full_path, "rb"))
        print(len(orig_data), attr_logits.shape[0])
        assert len(orig_data) == attr_logits.shape[0], "Number of samples in original data does not match extracted concepts"

        for i in range(len(orig_data)):
            orig_data[i]["attribute_label"] = attr_logits[i]
       
        if args.output_dir is None:
            save_path = args.data_dir + "extracted_concepts_" + args.model_type + "/"
        else:
            save_path = args.data_dir + args.output_dir + "/"
        os.makedirs(save_path, exist_ok=True)
        with open(save_path + pkl_file, "wb") as f:
            pickle.dump(orig_data, f)

                
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract concepts from trained model')
    parser.add_argument('-model_path', default=None, help='path to the trained model to extract concepts from')
    parser.add_argument('-pretrained', default=None, help='pretrained model to use for the X -> C part of the model')
    parser.add_argument('-n_attributes', type=int, default=112, help='number of attributes to predict')
    parser.add_argument('-n_class_attr', type=int, default=2, help='whether attr prediction is a binary or triary classification')
    parser.add_argument('-data_dir', default='Data/', help='directory to the root data')
    parser.add_argument('-pkl_file_dir', default='class_attr_data_10/', help='directory to the CUB pkl files relative to data_dir')
    parser.add_argument('-use_attr', help='whether to load attribute labels from the pkl files', action='store_true')
    parser.add_argument('-no_img', help='whether to load images (instead of just attributetes) from the pkl files', action='store_true')
    parser.add_argument('-batch_size', type=int, default=64, help='batch size for loading data and extracting concepts')
    parser.add_argument('-cub_data_dir', default='CUB_200_2011/', help='directory to the CUB image data') 
    parser.add_argument('-model_type', default='ModelXtoC', help='type of model to extract concepts from, needed to determine how to load the model and what results to return.')
    parser.add_argument('-output_dir', help='directory to save the extracted concepts to', default=None)
    
    
    args = parser.parse_args()
    extract_concepts(args)
    
