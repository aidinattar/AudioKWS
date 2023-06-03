"""

Parameters
----------

main_path : str
    main path to the dataset

epochs : int, optional
    number of epochs to train 

lr : float, optional
    learning rate

dropout : float, optional
    dropout rate

bidirectional : bool, optional
    whether to use bidirectional LSTM

attention : bool, optional
    whether to use attention

bidirectional : bool, optional
    whether to use bidirectional LSTM

batch_size : int, optional
    batch size

load_pretrained : bool, optional
    whether to load pretrained model

save_dir : str, optional
    directory to save checkpoints

ndata_train : int, optional
    number of data to train (default: None, all data)

ndata_test : int, optional
    number of data to train (default: None, all data)

seed : int, optional
    random seed (default: 1)

"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn as nn
import os
import glob
import copy
import pandas as pd
import numpy as np
from WaveformDataset import WaveformDataset


import argparse

parser = argparse.ArgumentParser(description='training')
parser.add_argument('--main_path', type=str, default='waveform_ds', help='main path to the dataset')
parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate ')
parser.add_argument('--dropout', type=float, default=0.4, help='dropout rate')
parser.add_argument('--batch_size', type=int, default=128, help='batch size (default: 256)')
parser.add_argument('--load_pretrained', type=bool, default=False, help='whether to load pretrained model')
parser.add_argument('--save_dir', type=str, default='checkpoints_no_attention', help='directory to save checkpoints')
parser.add_argument('--attention', type=bool, default=False, help='whether to use attention')
parser.add_argument('--bidirectional', type=bool, default=False, help='whether to use bidirectional LSTM')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--model', type=str, default='LSTM', help='model to use (default: LSTM)', choices=['LSTM', 'CNN'])

args = parser.parse_args()

def main(
    main_path,
    epochs=40,
    lr=1e-3,
    dropout=0.4,
    batch_size=64,
    load_pretrained=False,
    save_dir='checkpoints',
    attention=True,
    bidirectional=True,
    seed = 1,
    model='LSTM'
    ):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.manual_seed(seed)
    # load data
    from torch.utils.data import DataLoader

    print ('Loading training data...')
    train_dataset = WaveformDataset(main_path, data_type='train', )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4) 
    print ('Loading validation data...')
    val_dataset = WaveformDataset(main_path, data_type='val')
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


    nlabels = train_dataset.get_n_classes()
    idx2label = train_dataset.get_idx2label()

    # save to file get_idx2label
    with open(os.path.join(save_dir, 'idx2label.txt'), 'w') as f:
        for idx, label in idx2label.items():
            f.write(f'{idx} {label}\n')

    if model == 'LSTM':
        from LSTMmodel import LSTMmodel, LSTMModelMetaData
        # test the model
        hidden_size = 200
        model = LSTMmodel(
                        hidden_size=hidden_size, 
                        num_classes=nlabels, 
                        dropout=dropout, 
                        attention=attention,
                        bidirectional=bidirectional,
                        )
        meta = LSTMModelMetaData(
                        hidden_size=hidden_size, 
                        num_classes=nlabels, 
                        dropout=dropout, 
                        attention=attention,
                        bidirectional=bidirectional,
                        )
    elif model == 'CNN':
        from CNNmodel import CNNModel, CNNModelMetaData
        model = CNNModel(
                        in_channels=1, 
                        n_classes=nlabels,
                        )
        meta = CNNModelMetaData(
                        in_channels=1,
                        n_classes=nlabels,
                        )

    meta.save(os.path.join(save_dir, 'model_meta.json'))

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # loss function
    criterion = nn.CrossEntropyLoss()

    # early stopping
    from EarlyStopping import EarlyStopping
    early_stopping = EarlyStopping(patience=5, )

    # load pretrained model as a second learning strategy
    if load_pretrained:
        print ("Loading pretrained model...")
        models_saved = glob.glob(os.path.join(save_dir, 'model_*.torch'))
        if len(models_saved) > 0:
            # get most recent model
            epoches_done = max([int(model.split('_')[-1].split('.')[0]) for model in models_saved])
            model_path = os.path.join(save_dir, f'model_{epoches_done}.torch')
            print(f"Loading model from {model_path}")
            model.load_state_dict(torch.load(model_path))
        else:
            print("No model saved to load from")
            load_pretrained = False
            epoches_done = 0
    else: epoches_done = 0

   
    #  device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print ("device = ", device)
    model.to(device)

    # classifier
    from classifier import classifier
    classifier = classifier(model, device=device, )

    # train or test
    print ("Starting training")
    import time
    start_time = time.time()

    classifier.train(
                    train_loader, 
                    val_loader, 
                    epochs=epochs, 
                    optimizer=optimizer, 
                    loss_fn=criterion, 
                    save_dir=save_dir, 
                    start_epoch=epoches_done+1,
                    save_every=10,
                    early_stopping=early_stopping,
                    )

    
    time_elapsed = (time.time() - start_time) / 3600

    print ("Training done, time elapsed = ", time_elapsed, " hours")


if __name__ == '__main__':
    main(
        **vars(args)
        )
    