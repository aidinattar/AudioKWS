import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):

    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int,
                 stride: int):
        super().__init__()

        self.conv = nn.Conv1d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding='same'
                                )
        
        self.bn = nn.BatchNorm1d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    

class CNNModel(nn.Module):
    """
    From https://arxiv.org/abs/1909.04939

    Args:
        in_channels (int): number of input channels
        n_classes (int): number of classes to predict

    """

    def __init__(self, 
                 in_channels: int, 
                 n_classes: int = 1
                 ):
        super().__init__()

        self.conv1 = ConvBlock(in_channels, 128, 64, 1)
        self.conv2 = ConvBlock(128, 128, 16, 1)
        self.conv3 = ConvBlock(128, 128, 4, 1)
        self.final = nn.Linear(128, n_classes)

    def forward(self, x):
        # add channel dimension
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.final(x.mean(dim=-1))


class CNNModelMetaData():

    def __init__(self, 
                 in_channels: int, 
                 n_classes: int = 1
                 ):
        self.in_channels = in_channels
        self.n_classes = n_classes

    def __str__(self):
        return dict(in_channels=self.in_channels,
                    n_classes=self.n_classes).__str__()
    
    def save(self, path):
        with open(path, 'w') as f:
            f.write(str(self))


if __name__ == '__main__':
    from WaveformDataset import WaveformDataset
    from torch.utils.data import DataLoader

    dataset = WaveformDataset('waveform_ds', data_type='test')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    n_classes = dataset.get_n_classes()
    print ('n_classes',n_classes)

    model = CNNModel(in_channels=1, n_classes=n_classes)
    print (model)

    for batch in dataloader:
        x, y = batch
        out = model(x)
        print (out.shape)
        break
