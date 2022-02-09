import torch.nn.functional as F
import torch
import torch.nn as nn
import pdb

def nll_loss(output, target, *args):
    return F.nll_loss(output, target)

def logcosh(output, target, *args):
    x = output - target
    return torch.mean(x + torch.nn.Softplus()(-2*x) - torch.log(torch.tensor(2.)))

def MAE(output, target, *args):
    return F.l1_loss(output, target)

def MSE(output, target, *args):
    return F.mse_loss(output, target)

def crossentropy_loss(output, target, *args):
    return nn.CrossEntropyLoss()(output, target)

def bce_loss(output, target, *args):
    return nn.BCELoss()(output, target.float())

def trace_nll_loss(output, target, *args):
    target = target.squeeze()
    output = output.squeeze()
    return -torch.trace(
                torch.matmul(target, torch.log(output+1e-10).float().t()) +
                torch.matmul((1-target), torch.log(1-output+1e-10).float().t())
            ) / output.size(0)

def WordPTB_crossentropy_loss(output, target, *args):
    final_output = output[:, 40:].contiguous().view(-1, 10000)
    final_target = target[:, 40:].contiguous().view(-1) 
    return nn.CrossEntropyLoss()(final_output, final_target)

def weighted_L1_loss(output, target, *args):
    num_batches = args[0]
    eps = (.001 / num_batches)
    target = target.float()
    return (F.l1_loss(output, target) * (target + target.mean() + eps)).mean()
