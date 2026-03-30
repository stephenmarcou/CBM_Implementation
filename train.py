import torch 
from utils import accuracy, Logger, AverageMeter, binary_accuracy
import os
from config import CUB_DATA_DIR, PKL_FILE_DIR, MIN_LR, LR_DECAY_SIZE, PKL_FILE_INCOMPLETE_DIR, N_CLASSES, ROOT_LOG_DIR, DATA_DIR
from models import ModelCtoy, ModelXtoCtoY
from dataset import load_data, find_class_imbalance, create_incomplete_concept_data
import math


if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
    
def run_epoch_from_raw_input(model, optimizer, loader, loss_meter, acc_meter, criterion, attr_criterion, args, is_training):
    """
    For the rest of the networks (X -> A, cotraining, simple finetune)
    """
    if is_training:
        model.train()
    else:
        model.eval()

    for batch_idx, data in enumerate(loader):
        if attr_criterion is None:
            inputs, labels = data
            attr_labels = None
        # image and attribute used for prediction 
        else:
            inputs, labels, attr_labels = data
            if args.n_attributes > 1:
                attr_labels = [i.long() for i in attr_labels]
                attr_labels = torch.stack(attr_labels).t() #shape (batch_size, n_attributes)
            else:
                if isinstance(attr_labels, list):
                    attr_labels = attr_labels[0]
                attr_labels = attr_labels.unsqueeze(1)
                

            attr_labels = attr_labels.to(device)

        inputs = inputs.to(device)
        labels = labels.to(device)


        class_outputs, attr_outputs = model(inputs)

        losses = []
        if not args.bottleneck:
            loss_main = criterion(class_outputs, labels)
            losses.append(loss_main)
        if attr_criterion is not None and args.attr_loss_weight > 0: #X -> A, cotraining, end2end
            for i in range(len(attr_criterion)):
                losses.append(
                                args.attr_loss_weight * attr_criterion[i](
                                    attr_outputs[:, i].float(),
                                    attr_labels[:, i].float()
                                )
                            )


        if args.bottleneck: #attribute accuracy
            sigmoid_outputs = torch.sigmoid(attr_outputs)
            acc = binary_accuracy(sigmoid_outputs, attr_labels)
            acc_meter.update(acc.data.cpu().numpy(), inputs.size(0))
        else:
            acc = accuracy(class_outputs, labels, topk=(1,)) #only care about class prediction accuracy
            acc_meter.update(acc[0], inputs.size(0))
            
            
            # if batch_idx == 30:
            #     predicted_classes = torch.argmax(class_outputs, dim=1)
            #     print(f"Predicted classes: {predicted_classes}")
            #     print(f"True classes: {labels}")
                
       
        #print(f"length losses: {len(losses)}")
        if attr_criterion is not None:
            if args.bottleneck:
                total_loss = sum(losses)/ args.n_attributes
            else: #cotraining, loss by class prediction and loss by attribute prediction have the same weight
                total_loss = losses[0] + sum(losses[1:])
                if args.normalize_loss:
                    total_loss = total_loss / (1 + args.attr_loss_weight * args.n_attributes)
       
        else: #finetune
            total_loss = sum(losses)
        loss_meter.update(total_loss.item(), inputs.size(0))
        if is_training:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
    return loss_meter, acc_meter
    
    

    

def run_epoch_c_to_y(model, optimizer, loader, loss_meter, acc_meter, criterion, is_training):
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

    # Ensure all models go into the same log dir
    full_path_log_dir = ROOT_LOG_DIR + args.log_dir
    
    # Log
    if os.path.exists(full_path_log_dir): # job restarted by cluster
        for f in os.listdir(full_path_log_dir):
            os.remove(os.path.join(full_path_log_dir, f))
    else:
        os.makedirs(full_path_log_dir)
    
    log_file_name = args.exp + "_log.txt"
    logger = Logger(os.path.join(full_path_log_dir, log_file_name))
    logger.write('\n' + str(args) + '\n')
    # logger.write(str(imbalance) + '\n') Need to be impemented later
    logger.flush()
    
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    

    # Determine imbalance
    imbalance = None
    if args.use_attr and not args.no_img and args.weighted_loss:
        train_data_path = DATA_DIR + PKL_FILE_DIR + 'train.pkl'
        if args.weighted_loss == 'multiple':
            imbalance = find_class_imbalance(train_data_path, multiple_attr=True)
        else:
            imbalance = find_class_imbalance(train_data_path, multiple_attr=False)
    
    # Use attributes and raw images for class prediction
    if args.use_attr and not args.no_img:
        attr_criterion = [] 
        # use imbalance ratio to weight the loss for positive samples for each attribute if -weighted_loss flag is included, 
        # otherwise use unweighted loss for each attribute
        if args.weighted_loss:
            assert(imbalance is not None)
            for ratio in imbalance:
                 # weighted: w*BCE(x,y) = w*[-y*log(sigmoid(x)) - (1-y)*log(1-sigmoid(x))]
                attr_criterion.append(torch.nn.BCEWithLogitsLoss(weight=torch.FloatTensor([ratio]).to(device))) # stephen changed from weight to pos_weight
        else:
            for i in range(args.n_attributes):
                attr_criterion.append(torch.nn.CrossEntropyLoss())
    else:
        attr_criterion = None
    
    
    
    
    
    # Optimizer setup
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "RMSprop":
        optimizer = torch.optim.RMSprop((p for p in model.parameters() if p.requires_grad), lr=args.lr,
                            weight_decay=args.weight_decay, momentum=args.momentum)
    else:
        optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=args.lr, 
                        weight_decay=args.weight_decay, momentum=args.momentum)
    
    # Reduces learning rate by a factor of 10 every args.scheduler_step epochs until it reaches MIN_LR
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=0.1)
    num_epoch_till_min_LR = int(math.log(MIN_LR / args.lr) / math.log(LR_DECAY_SIZE)) * args.scheduler_step
    
    
    
    # Train on incomplete set of concept data if -incomplete flag is included, otherwise train on complete set of concept data
    if args.incomplete:
        # Check if incomplete data files exist, otherwise create them
        create_incomplete_concept_data(args.n_attributes) # creates incomplete concept data and saves it to pkl file
        train_data_path = os.path.join(DATA_DIR, PKL_FILE_INCOMPLETE_DIR, 'train.pkl')
        val_data_path = train_data_path.replace('train.pkl', 'val.pkl')
    else:
        train_data_path = os.path.join(DATA_DIR, PKL_FILE_DIR, 'train.pkl')
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
            train_loss_meter, train_acc_meter = run_epoch_c_to_y(model, optimizer,
                                                                       train_loader, train_loss_meter, train_acc_meter, 
                                                                       criterion, is_training=True)
        else:
            run_epoch_from_raw_input(model, optimizer, train_loader, train_loss_meter, train_acc_meter, criterion, attr_criterion, args, is_training=True)
        
        # If not retraining, evaluate on validation set at end of each epoch and save best model
        if not args.ckpt:
            val_loss_meter = AverageMeter()
            val_acc_meter = AverageMeter()

            with torch.no_grad():
                if args.no_img:
                    val_loss_meter, val_acc_meter = run_epoch_c_to_y(model, optimizer,
                                                                           val_loader, val_loss_meter, val_acc_meter, 
                                                                           criterion, is_training=False)
                else:
                    val_loss_meter, val_acc_meter = run_epoch_from_raw_input(model, optimizer,
                                                                             val_loader, val_loss_meter, val_acc_meter,
                                                                             criterion, attr_criterion, args, is_training=False)

        # If retraining
        else: 
            val_loss_meter = train_loss_meter
            val_acc_meter = train_acc_meter
            
        train_loss_avg = train_loss_meter.avg
        val_loss_avg = val_loss_meter.avg
        
        if best_val_acc < val_acc_meter.avg:
            best_epoch = epoch
            best_val_acc = val_acc_meter.avg
            save_file = "best_model_" + args.exp + ".pt"
            torch.save(model.state_dict(), os.path.join(ROOT_LOG_DIR, args.log_dir, save_file))
            
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
    model = ModelCtoy(pretrained=args.pretrained, freeze=args.freeze, input_dim=args.n_attributes, output_dim=N_CLASSES, expand_dim=args.expand_dim)
    train(model, args)
    
def train_joint(args):
    model = ModelXtoCtoY(n_class_attr=args.n_class_attr, pretrained=args.pretrained, num_classes=N_CLASSES, n_attributes=args.n_attributes, expand_dim=args.expand_dim,
                 use_relu=args.use_relu, use_sigmoid=args.use_sigmoid)
    train(model, args)
            
