import torch
from torch.utils.data import Dataset

class AnDetDataWrapper(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.data[idx]
        return data, label