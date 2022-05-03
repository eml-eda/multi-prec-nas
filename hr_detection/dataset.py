import numpy as np
from pathlib import Path
import pickle
import torch
from torch.utils.data import Dataset

SUBJ = 15 # Number of subjects in Dalia Dataset

def cross_validation(kfold_it):
    raise NotImplementedError()

def split_train_test(data_dir, rng, test_ratio=.1):
    # Load data
    data_dir = Path(data_dir)
    with open(data_dir / 'slimmed_dalia.pkl', 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    x, y, groups = data.values()

    # Split x,y in train and test
    x_train, y_train = list(), list()
    x_test, y_test = list(), list()
    for s in range(1, SUBJ+1):
        # Select current subject
        mask_subj = groups == s
        # Compute the `test_ratio`% samples for current subject
        test_len = int(len(x[mask_subj]) * test_ratio)
        # Sample `test_len` random indexes
        random_idx = rng.choice(len(x[mask_subj]), test_len, replace=False)
        # Build binary mask for selected samples
        mask_idx = np.zeros(len(x[mask_subj]), dtype=bool)
        mask_idx[random_idx] = True
        # Split data
        x_train.append(x[mask_subj][~mask_idx])
        y_train.append(y[mask_subj][~mask_idx])
        x_test.append(x[mask_subj][mask_idx])
        y_test.append(y[mask_subj][mask_idx])
    x_test = np.vstack(x_test)
    x_train = np.vstack(x_train)
    y_test = np.vstack(y_test)
    y_train = np.vstack(y_train)
    return DaliaDataset(x_train, y_train), DaliaDataset(x_test, y_test)

class DaliaDataset(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        return self.x[idx], self.y[idx]

# For testing purposes
if __name__ == '__main__':
    rng = np.random.default_rng(seed=42)
    split_train_test('data', rng)