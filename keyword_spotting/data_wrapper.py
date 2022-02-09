import torch
from torch.utils.data import Dataset

class KWSDataWrapper(Dataset):
    def __init__(self, data_generator):
        self.data_generator = data_generator
        self.min = -123.5967
        self.max = 43.6677
    
    def __len__(self):
        return len(self.data_generator)
    
    def __getitem__(self, idx):
        data = torch.from_numpy(self.data_generator[idx][0])
        #data_norm = (data - self.min) / (self.max - self.min)
        label = torch.from_numpy(self.data_generator[idx][1])
        #return data_norm, label
        return data, label