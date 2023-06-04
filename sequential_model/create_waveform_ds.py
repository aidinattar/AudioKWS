"""
Create waveform dataset for sequential model and save as txt files
This is useful to be later used in the torch dataloader
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dataset import *
from utils.input import *
import argparse
import numpy as np
import torch

parser = argparse.ArgumentParser(description='Create waveform dataset')
parser.add_argument('--create_files', action='store_true',
                    help='create files or simply load them', default=False)
parser.add_argument('--noisy', action='store_true',
                    help='augment data with noise', default=False)
parser.add_argument('--augment', action='store_true',
                    help='augment data with shift', default=False)

args = parser.parse_args()

if __name__ == '__main__':
    data = DataLoader(path='../DATA/speech_commands_v0.02')

    if args.create_files:
        filenames = data.get_filenames()
        train_files, val_files, test_files = data.split_data(filenames)

        # Create waveform dataset
        X_train, y_train = data.get_waveform_data(train_files)
        X_val, y_val = data.get_waveform_data(val_files)
        X_test, y_test = data.get_waveform_data(test_files)

        # save as txt
        folder = 'waveform_ds'

        # pad to max length
        max_len = max([len(x) for x in X_train] + [len(x) for x in X_val] + [len(x) for x in X_test])
        X_train = [np.pad(x, (0, max_len - len(x)), 'constant') for x in X_train]
        X_val = [np.pad(x, (0, max_len - len(x)), 'constant') for x in X_val]
        X_test = [np.pad(x, (0, max_len - len(x)), 'constant') for x in X_test]

    else:
        X_train = torch.load('waveform_ds/X_train.pt')
        X_val = torch.load('waveform_ds/X_val.pt')
        X_test = torch.load('waveform_ds/X_test.pt')
        y_train = np.loadtxt('waveform_ds/y_train.txt', dtype=str)
        y_val = np.loadtxt('waveform_ds/y_val.txt', dtype=str)
        y_test = np.loadtxt('waveform_ds/y_test.txt', dtype=str)

        from data_augmentation_utils import *

        if args.noisy and not args.augment:
            # create a noise dataset
            folder = 'waveform_ds_noisy'
            
            def random_noise(x):
                # random uniform in (0.001, 0.01)
                rate = np.random.uniform(0.001, 0.01)
                return add_noise(x, noise_factor=rate)
            
            np.random.seed(42)
            X_train_more = [random_noise(x) for x in X_train]
            X_val_more = [random_noise(x) for x in X_val]
            X_test_more = [random_noise(x) for x in X_test]

        elif args.augment and not args.noisy:
            # create a shift dataset
            folder = 'waveform_ds_augmented'

            def random_aug(x):
                # random uniform in (0.001, 0.01)
                which = np.random.choice([0, 1, 2])
                if which == 0:
                    # shift
                    return shift(x, shift_size=400)
                elif which == 1:
                    # stretch
                    return stretch(x, rate=np.uniform(0.9, 1.05))
                elif which == 2:
                    # pitch
                    return pitch_shift(x, pitch_factor=np.random.uniform(0.9, 1.3))
            
            np.random.seed(42)
            X_train_more = [random_aug(x) for x in X_train]
            X_val_more = [random_aug(x) for x in X_val]
            X_test_more = [random_aug(x) for x in X_test]

        elif args.augment and  args.noisy:
            # create a shift dataset
            folder = 'waveform_ds_all'

            def random_aug(x):
                # random uniform in (0.001, 0.01)
                which = np.random.choice([0, 1, 2, 3])
                if which == 0:
                    # shift
                    return shift(x, shift_size=400)
                elif which == 1:
                    # stretch
                    return stretch(x, rate=np.uniform(0.9, 1.05))
                elif which == 2:
                    # pitch
                    return pitch_shift(x, pitch_factor=np.random.uniform(0.9, 1.3))
                elif which == 3:
                    # noise
                    return add_noise(x, noise_factor=np.random.uniform(0.001, 0.01))
                

            np.random.seed(42)
            X_train_more = [random_aug(x) for x in X_train]
            X_val_more = [random_aug(x) for x in X_val]
            X_test_more = [random_aug(x) for x in X_test]

        X_train = list(X_train.numpy()) + list(X_train_more)
        X_val = list(X_val.numpy()) + list(X_val_more)
        X_test = list(X_test.numpy()) + list(X_test_more)


    if not os.path.exists(folder):
        os.makedirs(folder)

    X_test = torch.tensor(X_test)
    X_train = torch.tensor(X_train)
    X_val = torch.tensor(X_val)

    # save as torch tensor
    torch.save(X_test, f'{folder}/X_test.pt')
    torch.save(X_train, f'{folder}/X_train.pt')
    torch.save(X_val, f'{folder}/X_val.pt')

    with open(f'{folder}/y_test.txt', 'w') as f:
        for item in y_test:
            f.write("%s\n" % item)
    
    with open(f'{folder}/y_train.txt', 'w') as f:
        for item in y_train:
            f.write("%s\n" % item)

    with open(f'{folder}/y_val.txt', 'w') as f:
        for item in y_val:
            f.write("%s\n" % item)

    




   
