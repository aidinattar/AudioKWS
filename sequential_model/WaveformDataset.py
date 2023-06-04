import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd

class WaveformDataset(Dataset):
    """
    Dataset for waveform data
    """
    def __init__(self, folder_path, data_type='train', samples=15000, extract_features=True):
        """
        Args:
            folder_path (str): path to folder with X_train.pt, X_val.pt, X_test.pt, y_train.txt, y_val.txt, y_test.txt
            data_type (str): 'train', 'val' or 'test'
            samples (int): number of samples to extract from each waveform
            extract_features (bool): if True, extract features from raw data and use them as input
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

        print (f'X_{data_type}.shape: {self.X.shape}')
        # subsample to the min number of samples per clas
        label_count = pd.Series(self.y_names).value_counts()
        min_count = label_count.min()
        X, y = [], []
        names = []
        for label in self.label_to_idx.keys():
            idxs = np.where(self.y_names == label)[0]
            idxs = np.random.choice(idxs, min_count, replace=False)
            X.extend(self.X[idxs])
            y.extend(self.y[idxs])
            names.extend(self.y_names[idxs])

        self.X = torch.stack(X)
        self.y = torch.stack(y)
        self.y_names = np.array(names)
        print (f'X_{data_type}.shape: {self.X.shape}')

        # subsample to samples
        if samples is not None:
            from scipy.signal import resample
            X = []
            # map resample to each sample in X
            for sample in self.X:
                X.append(resample(sample, samples))
            self.X = torch.tensor(X)
            print(f'X_{data_type}.shape: {self.X.shape}')

        # normalize in (-1, 1)
        self.X = self.X / self.X.abs().max()

        # extract features
        if extract_features:
            self.X = self.extract_features(self.X)
            print(f'X_{data_type}.shape: {self.X.shape}')

    
    def extract_features(self, examples):
        from transformers import AutoFeatureExtractor
        feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

        audio_arrays = [x.numpy() for x in examples]
        inputs = feature_extractor(
            audio_arrays, sampling_rate=feature_extractor.sampling_rate, max_length=16000, truncation=True
        )
        inputs = inputs["input_values"]
        inputs = torch.tensor(inputs)
        return inputs


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

    # plot waveform
    import matplotlib.pyplot as plt
    plt.plot(dataset[10][0])
    plt.show()

    
