import torch
from torch.utils.data import Dataset

class KWSDataWrapper(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        #self.min = -123.5967
        #self.max = 43.6677
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        data = torch.from_numpy(self.x[idx])
        label = torch.from_numpy(self.y[idx])
        return data, label