"""
Parameters
----------

batch_size : int
    batch size

save_dir : str
    directory where checkpoints are saved. all results are saved in this directory

ndata : int
    number of data to train (default: None, all data)

data_path : str
    path to data

"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import glob
import json
import numpy as np
import pandas as pd
from WaveformDataset import WaveformDataset


import seaborn as sns
import matplotlib.pyplot as plt

import argparse
# baseline_lang_detection/syll
# dataset/cleaned/ngram_dataset/lang_syllables/
parser = argparse.ArgumentParser(description='training')
parser.add_argument('--batch_size', type=int, default=128, help='batch size (default: 128)')
parser.add_argument('--save_dir', type=str, default='checkpoints', help='directory to save checkpoints (default: checkpoint)')
parser.add_argument('--main_path', type=str, default='waveform_ds', help='main path to the dataset')
parser.add_argument('--model', type=str, default='CNN', help='model to use (default: LSTM)', choices=['LSTM', 'CNN'])

args = parser.parse_args()


def main(
    batch_size=64,
    save_dir='checkpoints',
    main_path='dataset',
    model='CNN',
    ):
    torch.manual_seed(1)

    # check that savedir exists
    if not os.path.exists(save_dir):
        raise Exception("No checkpoint found in {}".format(save_dir))

    # load model metadata
    #metadata are saved in the same directory as the model checkpoint
    # metadata_path = os.path.join(save_dir, 'model_meta.json')
    # # check if metadata exists
    # if not os.path.exists(metadata_path):
    #     raise Exception("No metadata found")

    # with open(metadata_path, 'r') as f:
    #     model_metadata = json.load(f)
    

    # load data
    from torch.utils.data import DataLoader

    print ('Loading training data...')
    train_dataset = WaveformDataset(main_path, data_type='train', )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1) 
    print ('Loading validation data...')
    test_dataset = WaveformDataset(main_path, data_type='test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    nlabels = train_dataset.get_n_classes()


    # plot learning curve
    from utils import plot_learning_curves, load_idx2label
    plot_learning_curves(
        val_path=os.path.join(save_dir, 'val_loss.npy'),
        train_path=os.path.join(save_dir, 'train_loss.npy'),
        save_dir=save_dir,
        )
    idx2label = load_idx2label(save_dir)


    # if model == 'LSTM':
    #     from LSTMmodel import LSTMmodel, LSTMModelMetaData
    #     # test the model
    #     hidden_size = 200
    #     model = LSTMmodel(
    #                     hidden_size=hidden_size, 
    #                     num_classes=nlabels, 
    #                     dropout=model_metadata['dropout'], 
    #                     attention=model_metadata['attention'],
    #                     bidirectional=model_metadata['bidirectional'],
    #                     )
        
    # elif model == 'CNN':
    #     from CNNmodel import CNNModel, CNNModelMetaData
    #     model = CNNModel(
    #                     in_channels=1, 
    #                     n_classes=nlabels,
    #                     )
    from CNNmodel import CNNModel, CNNModelMetaData
    model = CNNModel(
                    in_channels=1, 
                    n_classes=nlabels,
                    )

    #  device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print ("device = ", device)
    model.to(device)

    # classifier
    from classifier import classifier
    classifier = classifier(model, device=device,)
    
    # use best model if exists
    if os.path.exists(os.path.join(save_dir, 'best_model.torch')):
        print ("Loading best model")
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.torch')))
        else:
            model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.torch'), map_location=torch.device('cpu')))
    else:
        raise Exception("No best model saved")


    ##############################
    # test on train set
    ##############################
    print ("Starting testing on train set")
    r_train = classifier.test(train_loader)
    print ("Testing on train set done")


    ##############################
    # test on test set
    ##############################
    print ("Starting testing on test set")
    r_test = classifier.test(test_loader)
    print ("Testing done")


    ##############################
    # print results and metrics
    ##############################
    results = {
        'train': r_train,
        'test': r_test,
    }     

    for d, r in results.items():
        results = r['metrics']
        report = r['report']
        cm = r['cm']
        print ("*"*20)
        print ("Results on ", d, " set")

        # print results
        print (f"\nMetrics: ")
        print (results)

        # classification report
        print ("\nClassification report: ")
        # remove last three rows (accuracy, macro avg, weighted avg)
        report = report[:-3]
        label_names = [idx2label[int(idx)] for idx in report.index]
        report.index = label_names
        report = report.sort_values(by='f1-score', ascending=False)
        report = report.round(2)
        print (report)

        # save classification report
        report_path = os.path.join(save_dir, d+'_classification_report.csv')
        report.to_csv(report_path)


        # confusion matrix
        print ("\nSaving confusion matrix: ")

        #round values to 2 decimals for annot
        annot = np.round(cm,2)
        annot = annot.astype('str')
        annot[annot=='0.0']='0'

        fig, ax = plt.subplots(figsize=(18,14))
        sns.set_theme(style='white', font_scale=1.5, palette='PuBu')
        sns.heatmap(cm, cmap='PuBu', annot=annot, ax=ax, cbar=False, fmt='')
        ax.set_xticklabels(label_names, rotation=90, fontsize=20)
        ax.set_yticklabels(label_names,rotation=0, fontsize=20)
        # change index in cm to names
        fig.savefig(os.path.join(save_dir, d+'_confusion_matrix.pdf'), bbox_inches='tight')


        # add labels to the clustermap
        print ("Confusion matrix: ")
        cm.columns = label_names
        cm.index = label_names
        sns.clustermap(cm, cmap='PuBu', annot=annot, fmt='', cbar=False,figsize=(18,14)).savefig(os.path.join(save_dir, d+'_confusion_matrix_clustermap.pdf'), bbox_inches='tight')

        # save top2 and top3 accuracy
        top2_acc = r['top2_acc']
        top3_acc = r['top3_acc']
        print ("Top2 accuracy = ", top2_acc)
        print ("Top3 accuracy = ", top3_acc)
    
        # save top2 and top3 accuracy together with classification report in  d+'_results.csv'
        with open (os.path.join(save_dir, d+'_results.txt'), 'w') as f:
            # metrics 
            f.write("{:<15s} {:<10s}\n".format('metric', 'value'))
            f.write("{:<15s} {:<10s}\n".format('accuracy', str(results['accuracy'][0])))
            f.write("{:<15s} {:<10s}\n".format('recall',  str(results['recall'][0])))
            f.write("{:<15s} {:<10s}\n".format('precision',  str(results['precision'][0])))
            f.write("{:<15s} {:<10s}\n".format('f1',  str(results['f1'][0])))
            f.write("{:<15s} {:<10s}\n".format('top2_acc',  str(top2_acc)))
            f.write("{:<15s} {:<10s}\n".format('top3_acc',  str(top3_acc)))
            
            # classification report
            f.write(f"\n\nClassification report:\n")
            f.write ('{:<20s} {:<10s} {:<10s} {:<10s} {:<10s}\n'.format('label', 'precision', 'recall', 'f1-score', 'support'))
            for label, row in report.iterrows():
                f.write ('{:<20s} {:<10s} {:<10s} {:<10s} {:<10s}\n'.format(label, str(row['precision']), str(row['recall']), str(row['f1-score']), str(row['support'])))




if __name__ == '__main__':
    main(
        **vars(args)
        )