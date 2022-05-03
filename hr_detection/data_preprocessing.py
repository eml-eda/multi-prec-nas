import argparse
import numpy as np
from pathlib import Path
import pickle
import random
from skimage.util.shape import view_as_windows

random.seed(42)

def main(data_dir):
    data_dir = Path(data_dir)

    dataset = collect_data(data_dir)
    preprocess_data(dataset, data_dir)

def collect_data(data_dir):
    dataset = dict()
    num = list(range(1, 15+1))
    session_list = random.sample(num, len(num))
    for subj in session_list:
        with open(data_dir / ('S'+str(subj)) / ('S'+str(subj)+'.pkl'), 'rb') as f:
            subject = pickle.load(f, encoding='latin1')
        ppg = subject['signal']['wrist']['BVP'][::2].astype('float32')
        acc = subject['signal']['wrist']['ACC'].astype('float32')
        target = subject['label'].astype('float32')
        dataset[subj] = {
                'ppg': ppg,
                'acc': acc,
                'target': target
                }
    return dataset

def preprocess_data(dataset, data_dir):
    """
    Process data with a sliding window of size 'time_window' and overlap 'overlap'
    """
    fs = 32
    time_window = 8
    overlap = 2

    groups = list()
    signals = list()
    targets = list()
    
    for k in dataset:
        sig = np.concatenate(
                    (dataset[k]['ppg'], dataset[k]['acc']), 
                    axis=1
                    )
        sig = np.moveaxis(
                    view_as_windows(
                        sig,
                        (fs*time_window, 4),
                        fs*overlap
                        )[:, 0, :, :],
                    1, 2
                    )
        
        ### Normalization ###
        # Put normalization here if needed

        groups.append(np.full(sig.shape[0], k))
        signals.append(sig)
        targets.append(np.reshape(
            dataset[k]['target'],
            (dataset[k]['target'].shape[0], 1)
            ))

    groups = np.hstack(groups)
    X = np.vstack(signals)
    y = np.reshape(
            np.vstack(targets),
            (-1, 1)
            )
    
    dataset = {
            'X': X,
            'y': y,
            'groups': groups
            }

    with open(data_dir.parent / 'slimmed_dalia.pkl', 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pre-Process Data')
    parser.add_argument('data_dir', type=str, help='path to data directory')
    args = parser.parse_args()
    print(args)

    main(args.data_dir)