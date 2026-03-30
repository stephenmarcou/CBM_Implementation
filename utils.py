import torch
import os
import sys



if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:   
    device = torch.device("cpu")


class AverageMeter(object):
    """
    Computes and stores the average and current value
    As training progress batch by batch, we can update the average loss and accuracy 
    using this class
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """
    Computes the precision@k for the specified values of k
    output and target are Torch tensors
    Checks if the correct class is among the top k predicted classes for each sample in the batch
    Returns the percentage of samples for which the correct class is among the top k predictions
    """
    maxk = max(topk)
    batch_size = target.size(0)
    # output shape is (batch_size, num_classes), topk returns top k class indices per sample in batch
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    temp = target.view(1, -1).expand_as(pred)
    temp = temp.to(device)
    correct = pred.eq(temp)

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        # Convert to percentage of correct predictions in batch
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def binary_accuracy(output, target):
    """
    Computes accuracy for multi-label binary predictions.
    output: probabilities (after sigmoid)
    target: binary labels (0 or 1)
    """
    pred = (output >= 0.5).float()
    correct = (pred == target).float().sum()
    acc = correct / target.numel()
    return acc * 100




class Logger(object):
    """
    Log results to a file and flush() to view instant updates
    """

    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()
            
            
