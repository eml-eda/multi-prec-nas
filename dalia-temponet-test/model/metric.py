import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import math
import pdb

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def MAE(output, target):
    return F.l1_loss(output, target)

def WordPTB_perplexity(output, target):
    final_output = output[:, 40:].contiguous().view(-1, 10000)
    final_target = target[:, 40:].contiguous().view(-1)
    loss = CrossEntropyLoss()(final_output, final_target)
    return math.exp(loss)

def FRR(output, target):
    with torch.no_grad():
        #true_negative = (output[target == 0] < .5).sum()
        true_positive = (output[target == 1] > .5).sum()
        false_negative = (output[target == 1] < .5).sum()
        #false_positive = (output[target == 0] > .5).sum()
    #pdb.set_trace()
    prova = false_negative / (true_positive + false_negative)
    return false_negative / (true_positive + false_negative + 1e-10) 
    #return false_negative / len(target)

def FAR(output, target):
    with torch.no_grad():
        true_negative = (output[target == 0] < .5).sum()
        #true_positive = (output[target == 1] > .5).sum()
        #false_negative = (output[target == 1] < .5).sum()
        false_positive = (output[target == 0] > .5).sum()
    #pdb.set_trace()
    return false_positive / (false_positive + true_negative + 1e-10) 
    #return false_positive / len(target)

def binary_accuracy(output, target):
    with torch.no_grad():
        true_negative = (output[target == 0] < .5).sum()
        true_positive = (output[target == 1] > .5).sum()
        #false_negative = (output[target == 1] < .5).sum()
        #false_positive = (output[target == 0] > .5).sum()
    return (true_positive + true_negative) / len(target)