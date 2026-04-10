import torch 
from utils import accuracy, Logger, AverageMeter, binary_accuracy
import os
from config import CUB_DATA_DIR, PKL_FILE_DIR, MIN_LR, LR_DECAY_SIZE, PKL_FILE_INCOMPLETE_DIR, N_CLASSES, ROOT_LOG_DIR, DATA_DIR
from models import ModelCtoy, ModelXtoCtoY, ModelXtoC, ModelXtoChat_ChatToY
from dataset import load_data, find_class_imbalance
import math
import time

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
    
    
def run_epoch_from_raw_input(model, optimizer, loader, loss_meter, acc_meter, criterion, attr_criterion, args, is_training, attr_acc_meter=None):
    """
    For the rest of the networks (X -> A, cotraining, simple finetune)
    """
    if is_training:
        model.train()
    else:
        model.eval()

    for batch_idx, data in enumerate(loader):
        print(f"Processing batch {batch_idx}...")
        t0 = time.time()
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
        print("Done loading data")
        t1 = time.time()

        inputs = inputs.to(device)
        labels = labels.to(device)

        if args.exp == "Concept_XtoC":
            attr_outputs = model(inputs)
        else:
            class_outputs, attr_outputs = model(inputs)

        t2 = time.time()
        
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
        #print(f"attr_outputs.shape: {attr_outputs.shape}")
        #print(f"attr_labels.shape: {attr_labels.shape}")

        if args.bottleneck: #attribute accuracy
            sigmoid_outputs = torch.sigmoid(attr_outputs)
            acc = binary_accuracy(sigmoid_outputs, attr_labels)
            acc_meter.update(acc.data.cpu().numpy(), inputs.size(0))
        else:
            acc = accuracy(class_outputs, labels, topk=(1,)) #only care about class prediction accuracy
            acc_meter.update(acc[0], inputs.size(0))
            
            # Optional attribute accuracy tracking for joint / end2end models
            if (
                attr_acc_meter is not None
                and attr_labels is not None
                and attr_outputs is not None
            ):
                sigmoid_outputs = torch.sigmoid(attr_outputs)
                attr_acc = binary_accuracy(sigmoid_outputs, attr_labels)
                attr_acc_meter.update(attr_acc.data.cpu().numpy(), inputs.size(0))
            
            
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
        t3 = time.time()

        if batch_idx < 10:
            print(
                f"batch {batch_idx}: "
                f"load+prep={t1-t0:.3f}s, "
                f"forward={t2-t1:.3f}s, "
                f"backward+step={t3-t2:.3f}s"
            )
    return loss_meter, acc_meter, attr_acc_meter
    
    

    

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
    print("In train function", flush=True)

    # Ensure all models go into the same log dir
    if os.path.isabs(args.log_dir):
        full_path_log_dir = args.log_dir
    else:
        full_path_log_dir = os.path.join(ROOT_LOG_DIR, args.log_dir)
    full_path_log_dir = ROOT_LOG_DIR + args.log_dir
    print("1", flush=True)
    # Log
    if os.path.exists(full_path_log_dir):
        for f in os.listdir(full_path_log_dir):
            os.remove(os.path.join(full_path_log_dir, f))
    else:
        os.makedirs(full_path_log_dir)
    
    print("2", flush=True)
    log_file_name = args.exp + "_log.txt"
    logger = Logger(os.path.join(full_path_log_dir, log_file_name))
    print("3.5", flush=True)
    logger.write('\n' + str(args) + '\n')
    # logger.write(str(imbalance) + '\n') Need to be impemented later
    print("3.75", flush=True)
    logger.flush()
    
    print("3", flush=True)
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    
    print("4", flush=True)
    # Determine imbalance
    imbalance = None
    if args.use_attr and not args.no_img and args.weighted_loss:
        train_data_path = args.data_dir + args.pkl_file_dir + 'train.pkl'
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
    
    
    

    train_data_path = os.path.join(args.data_dir, args.pkl_file_dir, 'train.pkl')
    val_data_path = train_data_path.replace('train.pkl', 'val.pkl')
        
    logger.write(f"train_data_path: {train_data_path}\n")
    
    
    print("Going to load data...")
    if args.ckpt: #retraining
        train_loader = load_data(args, [train_data_path, val_data_path], args.use_attr, args.no_img, args.batch_size, args.uncertain_labels, image_dir=args.image_dir, \
                                 n_class_attr=args.n_class_attr, resampling=args.resampling)
        val_loader = None
    else:
        train_loader = load_data(args, [train_data_path], args.use_attr, args.no_img, args.batch_size, args.uncertain_labels, image_dir=args.image_dir, \
                                 n_class_attr=args.n_class_attr, resampling=args.resampling)
        val_loader = load_data(args, [val_data_path], args.use_attr, args.no_img, args.batch_size, image_dir=args.image_dir, n_class_attr=args.n_class_attr)

    
    
    # Training loop

    best_epoch = -1
    best_val_loss = float('inf')
    best_val_acc = 0.0
    
    for epoch in range(0, args.epochs):
        train_loss_meter = AverageMeter()
        train_acc_meter = AverageMeter()
        if args.print_attr_acc:
            train_attr_acc_meter = AverageMeter()  
        else:
            train_attr_acc_meter = None
        
        
        # split between cases if concept is input or image is input
        if args.no_img:
            train_loss_meter, train_acc_meter = run_epoch_c_to_y(model, optimizer,
                                                                       train_loader, train_loss_meter, train_acc_meter, 
                                                                       criterion, is_training=True)
        else:
            train_loss_meter, train_acc_meter, train_attr_acc_meter = run_epoch_from_raw_input(model, optimizer, train_loader, train_loss_meter, train_acc_meter, criterion, attr_criterion, args, attr_acc_meter=train_attr_acc_meter, is_training=True)
        
        # If not retraining, evaluate on validation set at end of each epoch and save best model
        if not args.ckpt:
            val_loss_meter = AverageMeter()
            val_acc_meter = AverageMeter()
            if args.print_attr_acc:
                val_attr_acc_meter = AverageMeter()
            else:
                val_attr_acc_meter = None

            with torch.no_grad():
                if args.no_img:
                    val_loss_meter, val_acc_meter = run_epoch_c_to_y(model, optimizer,
                                                                           val_loader, val_loss_meter, val_acc_meter, 
                                                                           criterion, is_training=False)
                else:
                    val_loss_meter, val_acc_meter, val_attr_acc_meter = run_epoch_from_raw_input(model, optimizer,
                                                                                                 val_loader, val_loss_meter, val_acc_meter,
                                                                                                 criterion, attr_criterion, args, attr_acc_meter=val_attr_acc_meter, is_training=False)

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
            if os.path.isabs(args.log_dir):
                full_path_log_dir = args.log_dir
            else:
                full_path_log_dir = os.path.join(ROOT_LOG_DIR, args.log_dir)
            torch.save(model.state_dict(), os.path.join(full_path_log_dir, save_file))
            
            
        log_line = (
        f"Epoch {epoch}\t "
        f"Train loss: {train_loss_avg:.4f}\t "
        f"Train acc: {train_acc_meter.avg.item():.2f}%\t "
        f"Val loss: {val_loss_avg:.4f}\t "
        f"Val acc: {val_acc_meter.avg.item():.2f}%\t "
    )

        if args.print_attr_acc and train_attr_acc_meter is not None:
            log_line += (
                f"Train attr acc: {train_attr_acc_meter.avg:.2f}%\t "
                f"Val attr acc: {val_attr_acc_meter.avg:.2f}%\t "
            )

        log_line += f"Best Val epoch: {best_epoch}\n"
        logger.write(log_line)
        logger.flush()
        #logger.write(f"""Epoch {epoch}\t Train loss: {train_loss_avg:.4f}\t Train acc: {train_acc_meter.avg.item():.2f}%\t Val loss: {val_loss_avg:.4f}\t Val acc: {val_acc_meter.avg.item():.2f}%\t Best Val epoch: {best_epoch} \n""")
        #logger.flush()
        
        
        if epoch <= num_epoch_till_min_LR:
            scheduler.step() #scheduler step to update lr at the end of epoch   
        if epoch % 10 == 0:
            print('Current lr:', scheduler.get_last_lr())
        
        
        if epoch >= 100 and val_acc_meter.avg < 3:
            print("Early stopping because of low accuracy")
            break
        if epoch - best_epoch >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch}: validation accuracy did not improve for {args.early_stop_patience} epochs")
            break
            
    

### Training Individual models ###
def train_c_to_y(args):
    model = ModelCtoy(pretrained=args.pretrained, freeze=args.freeze, input_dim=args.n_attributes, output_dim=N_CLASSES, expand_dim=args.expand_dim)
    train(model, args)

def train_X_to_C(args):
    model = ModelXtoC(pretrained=args.pretrained, output_dim=args.n_attributes)
    train(model, args)
    
# Sequential
def train_Chat_to_y_and_test_on_Chat(args):
    model = ModelXtoChat_ChatToY(n_class_attr=args.n_class_attr, n_attributes=args.n_attributes,
                                 num_classes=N_CLASSES, expand_dim=args.expand_dim)
    train(model, args)
    
def train_joint(args):
    model = ModelXtoCtoY(n_class_attr=args.n_class_attr, pretrained=args.pretrained, num_classes=N_CLASSES, n_attributes=args.n_attributes, expand_dim=args.expand_dim,
                 use_relu=args.use_relu, use_sigmoid=args.use_sigmoid)
    print("successfully created model", flush=True)
    train(model, args)
    

            
