import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import RobustScaler, QuantileTransformer
import seaborn as sns
import torch

from data_wrapper import KWSDataWrapper
import get_dataset as kws_data
import kws_util

def main(data_dir):
    num_classes = 12
    #data_dir = args.data.parent.parent.parent / 'GoogleSpeechCommands'

    Flags, unparsed = kws_util.parse_command()
    Flags.data_dir = str(data_dir)
    Flags.bg_path = str(data_dir)
    Flags.batch_size = 1
    print(f'We will download data to {Flags.data_dir}')
    ds_train, ds_test, ds_val = kws_data.get_training_data(Flags)
    print("Done getting data")

    train_shuffle_buffer_size = 85511
    val_shuffle_buffer_size = 10102
    test_shuffle_buffer_size = 4890

    fig, ax = plt.subplots(1, 3, figsize=(10, 10))

    ds_train = np.stack([_[0].flatten() 
        for _ in ds_train.shuffle(train_shuffle_buffer_size).as_numpy_iterator()
        ], axis=0).flatten()
    sns.histplot(ds_train, kde=True, ax=ax[0])

    ds_val = np.stack([_[0].flatten()
        for _ in ds_val.shuffle(val_shuffle_buffer_size).as_numpy_iterator()
        ], axis=0).flatten()
    sns.histplot(ds_val, kde=True, ax=ax[1])

    ds_test = np.stack([_[0].flatten()
        for _ in ds_test.shuffle(test_shuffle_buffer_size).as_numpy_iterator()
        ], axis=0).flatten()
    sns.histplot(ds_test, kde=True, ax=ax[2])

    plt.savefig('keyword_spotting/data_stats.png')

def plot_norm_data(data_dir):
    num_classes = 12
    #data_dir = args.data.parent.parent.parent / 'GoogleSpeechCommands'

    Flags, unparsed = kws_util.parse_command()
    Flags.data_dir = str(data_dir)
    Flags.bg_path = str(data_dir)
    Flags.batch_size = 1
    print(f'We will download data to {Flags.data_dir}')
    ds_train, ds_test, ds_val = kws_data.get_training_data(Flags)
    print("Done getting data")

    train_shuffle_buffer_size = 85511
    val_shuffle_buffer_size = 10102
    test_shuffle_buffer_size = 4890

    x_train = np.stack([_[0].squeeze(0) 
            for _ in ds_train.shuffle(train_shuffle_buffer_size).as_numpy_iterator()
            ], axis=0)
    #
    fig, ax = plt.subplots()
    sns.histplot(x_train.ravel(), kde=True)
    ax.set_ylim(0, 160000)
    ax.title.set_text('Raw Training data')
    plt.savefig('keyword_spotting/raw_train.png')
    #
    inp_shape = x_train.shape[1:]
    x_train = QuantileTransformer().fit_transform(
        np.expand_dims(x_train.ravel(), -1)
        ).reshape(-1, *inp_shape)
    #
    fig, ax = plt.subplots()
    sns.histplot(x_train.ravel(), kde=True)
    ax.set_ylim(0, 160000)
    ax.title.set_text('QuantileTransformer Training data')
    plt.savefig('keyword_spotting/quantiletransf_train.png')
    #

if __name__ == '__main__':
    data_dir = '/space/risso/GoogleSpeechCommands'
    #main(data_dir)
    plot_norm_data(data_dir)