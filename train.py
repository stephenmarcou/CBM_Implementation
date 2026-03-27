import torch 
from utils import accuracy, Logger, AverageMeter
import os
from config import PKL_FILE_DIR, MIN_LR, LR_DECAY_SIZE
from models import ModelCtoy
from dataset import load_data
import math


if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    

def run_epoch_concept_to_y(model, optimizer, loader, loss_meter, acc_meter, criterion, is_training):
    if is_training:
        model.train()
    else:
        model.eval()
    
    for _, data in enumerate(loader):
        # Inputs will be list of tensors of size 64 (batch size) and length of list is number of attributes 
        inputs, labels = data
        if isinstance(inputs, list):
            # Convert list of tensors to a single tensor by stacking and transposing
            inputs = torch.stack(inputs).t().float()

        inputs = torch.flatten(inputs, start_dim=1).float()
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        acc = accuracy(outputs, labels)
        loss_meter.update(loss.item(), inputs.size(0))
        acc_meter.update(acc[0], inputs.size(0))
        if is_training:
            optimizer.zero_grad() #zero the gradients before backpropagation
            loss.backward()
            optimizer.step() #optimize the model parameters based on the computed gradients
    return loss_meter, acc_meter


def train(model, args):
    
    # Log
    if os.path.exists(args.log_dir): # job restarted by cluster
        for f in os.listdir(args.log_dir):
            os.remove(os.path.join(args.log_dir, f))
    else:
        os.makedirs(args.log_dir)
    
    log_file_name = args.exp + "_log.txt"
    logger = Logger(os.path.join(args.log_dir, log_file_name))
    logger.write(str(args) + '\n')
    # logger.write(str(imbalance) + '\n') Need to be impemented later
    logger.flush()
    
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    attr_criterion = None # Add later for args.use_attr and not args.no_img
    
    # Optimizer setup
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "RMSprop":
        optimizer = torch.optim.RMSprop((p for p in model.parameters() if p.requires_grad), lr=args.lr,
                            weight_decay=args.weight_decay, momentum=0.9)
    else:
        optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=args.lr, 
                        weight_decay=args.weight_decay)
    
    # Reduces learning rate by a factor of 10 every args.scheduler_step epochs until it reaches MIN_LR
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=0.1)
    num_epoch_till_min_LR = int(math.log(MIN_LR / args.lr) / math.log(LR_DECAY_SIZE)) * args.scheduler_step
    
    
    train_data_path = os.path.join(PKL_FILE_DIR, 'train.pkl')
    val_data_path = train_data_path.replace('train.pkl', 'val.pkl')
    logger.write(f"train_data_path: {train_data_path}\n")
    
    
    
    if args.ckpt: #retraining
        train_loader = load_data([train_data_path, val_data_path], args.use_attr, args.no_img, args.batch_size, args.uncertain_labels, image_dir=args.image_dir, \
                                 n_class_attr=args.n_class_attr, resampling=args.resampling)
        val_loader = None
    else:
        train_loader = load_data([train_data_path], args.use_attr, args.no_img, args.batch_size, args.uncertain_labels, image_dir=args.image_dir, \
                                 n_class_attr=args.n_class_attr, resampling=args.resampling)
        val_loader = load_data([val_data_path], args.use_attr, args.no_img, args.batch_size, image_dir=args.image_dir, n_class_attr=args.n_class_attr)

    
    
    # Training loop

    best_epoch = -1
    best_val_loss = float('inf')
    best_val_acc = 0.0
    
    for epoch in range(0, args.epochs):
        train_loss_meter = AverageMeter()
        train_acc_meter = AverageMeter()
        
        # split between cases if concept is input or image is input
        if args.no_img:
            train_loss_meter, train_acc_meter = run_epoch_concept_to_y(model, optimizer,
                                                                       train_loader, train_loss_meter, train_acc_meter, 
                                                                       criterion, is_training=True)
        else:
            raise NotImplementedError("Image input not implemented yet")
        
        # If not retraining, evaluate on validation set at end of each epoch and save best model
        if not args.ckpt:
            val_loss_meter = AverageMeter()
            val_acc_meter = AverageMeter()

            with torch.no_grad():
                if args.no_img:
                    val_loss_meter, val_acc_meter = run_epoch_concept_to_y(model, optimizer,
                                                                           val_loader, val_loss_meter, val_acc_meter, 
                                                                           criterion, is_training=False)
                else:
                    raise NotImplementedError("Image input not implemented yet")
                


        # If retraining
        else: 
            val_loss_meter = train_loss_meter
            val_acc_meter = train_acc_meter
            
        train_loss_avg = train_loss_meter.avg
        val_loss_avg = val_loss_meter.avg
        
        if best_val_acc < val_loss_avg:
            best_epoch = epoch
            best_val_acc = val_loss_avg
            save_file = "best_model_" + args.exp + ".pt"
            torch.save(model.state_dict(), os.path.join(args.log_dir, save_file))
            
        logger.write(f"""Epoch {epoch}\t Train loss: {train_loss_avg:.4f}\t Train acc: {train_acc_meter.avg.item():.2f}%\t Val loss: {val_loss_avg:.4f}\t Val acc: {val_acc_meter.avg.item():.2f}%\t Best Val epoch: {best_epoch} \n""")
        logger.flush()
        
        
        if epoch <= num_epoch_till_min_LR:
            scheduler.step() #scheduler step to update lr at the end of epoch   
        if epoch % 10 == 0:
            print('Current lr:', scheduler.get_last_lr())
        
        
        if epoch >= 100 and val_acc_meter.avg < 3:
            print("Early stopping because of low accuracy")
            break
        if epoch - best_epoch >= 100:
            print("Early stopping because acc hasn't improved for a long time")
            break
            
    

### Training Individual models ###
def train_c_to_y(args):
    model = ModelCtoy(pretrained=args.pretrained, freeze=args.freeze, input_dim=args.n_attributes, output_dim=args.n_attributes, expand_dim=args.expand_dim)
    train(model, args)
            
