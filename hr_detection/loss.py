import torch
import torch.nn as nn

class LogCoshLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, output, target):
        x = output - target
        return torch.mean(x + nn.Softplus()(-2.*x) - torch.log(torch.tensor(2.)))