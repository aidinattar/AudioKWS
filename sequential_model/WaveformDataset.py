import torch
import numpy as np
from torch.utils.data import Dataset


class WaveformDataset(Dataset):
    """
    Dataset for waveform data
    """
    def __init__(self, folder_path, data_type='train'):
        """
        Args:
            folder_path (str): path to folder with X_train.pt, X_val.pt, X_test.pt, y_train.txt, y_val.txt, y_test.txt
            data_type (str): 'train', 'val' or 'test'
        """
        if data_type not in ['train', 'val', 'test']:
            raise ValueError('data_type must be "train", "val" or "test"')
        
        self.folder_path = folder_path
        self.data_type = data_type
        self.X = torch.load(f'{folder_path}/X_{data_type}.pt')
        self.y_names = np.loadtxt(f'{folder_path}/y_{data_type}.txt', dtype=str)

        self.label_to_idx = {label: idx for idx, label in enumerate(np.unique(self.y_names))}
        self.idx_to_label = {idx: label for idx, label in enumerate(np.unique(self.y_names))}
        self.y = np.array([self.label_to_idx[label] for label in self.y_names])
        self.y = torch.tensor(self.y)
        self.n_classes = len(self.label_to_idx)

    def get_n_classes(self):
        return self.n_classes
    
    def get_idx2label(self):
        return self.idx_to_label

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    


if __name__ == '__main__':
    dataset = WaveformDataset('waveform_ds', data_type='train')
    print(dataset[0])
    print(dataset[0][0].shape)
    print(dataset[0][1].shape)
    print(dataset[0][1])
    print(dataset.label_to_idx)
    
