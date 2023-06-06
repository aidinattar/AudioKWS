#save metadata
import json
import os
import numpy as np
from tqdm import tqdm
import torch
import seaborn as sns
import matplotlib.pyplot as plt

    
        

def save_label(
        idx2label,
        save_dir: str,
    ):
    """
    save the idx2label dictionary in a txt file
    Parameters
    ----------
    idx2label : dict
        dictionary with index to label mapping
    save_dir : str
        directory to save the idx2label dictionary

    """
    # save in a form char: index
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open (os.path.join(save_dir, 'label.txt'), 'w') as f:
        for idx, label in idx2label.items():
            f.write(f'{label}\n')
    

def load_idx2label(
        path: str,
    ):
    """
    load the idx2label dictionary from a txt file
    Parameters
    ----------
    path : str
        path to the txt file

        
    Returns
    -------
    idx2label : dict
        dictionary with index to label mapping

    """
    path = os.path.join(path, 'idx2label.txt')
    idx2label = {}
    with open (path, 'r') as f:
        for idx, label in enumerate(f):
            idx2label[idx] = label.strip().split(' ')[1]
    return idx2label


            
def save_metadata(
        metadata,
        save_dir: str,

    ):
    # save metadata
    import json
    import os
    
    #create the directory if it does not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # save metadata to json
    dir_path = os.path.join(save_dir, 'metadata.json')
    
    with open(dir_path, 'w' ) as f:
        json.dump(metadata.__dict__, f, indent=4)

    
def plot_learning_curves(val_path, train_path, save_dir):
    sns.set_theme(style="white", palette="colorblind", color_codes=True, font_scale=1.5)

    print ("Plotting learning curves")
    # load data from file npy
    val = np.load(val_path)
    train = np.load(train_path)
    print (val.shape, train.shape)

    # plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.lineplot(x=range(len(val)), y=val, label='Validation', color='darkred', linewidth=2, ax=ax)
    sns.lineplot(x=range(len(train)), y=train, label='Training', palette='navy', linewidth=2, ax=ax)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Learning Curves')
    print ("Saving learning curves to ", save_dir)
    save_dir = os.path.join(save_dir, 'learning_curves.pdf')
    plt.savefig(save_dir, dpi=300, bbox_inches='tight')




def load_MetaData(
        path: str,
        model_type: str,
    ):
    """
    load the LSTM metadata from a json file
    Parameters
    ----------
    path : str
        path to the json file

        
    Returns
    -------
    metadata : dict
        dictionary with metadata
    model_type : str
        type of model

    """
    path = os.path.join(path, 'model_metadata.json')
    with open (path, 'r') as f:
        metadata = json.load(f)

    if model_type == 'LSTM':
        from LSTMmodel import LSTMModelMetaData
        metadata = LSTMModelMetaData(**metadata)
    elif model_type == 'CNN':
        from CNNmodel import CNNModelMetaData
        metadata = CNNModelMetaData(**metadata)

    return metadata
