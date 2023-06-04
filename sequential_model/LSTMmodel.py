import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class LSTMmodel(nn.Module):
    def __init__(self, hidden_size, num_classes=5, num_layers=2, attention=True, dropout=0.5, bidirectional=True):
        super().__init__()
        self.input_size = 1
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        
        self.bidirectional = bidirectional
        self.attention = attention
        if attention:
            self.attention_layer = nn.MultiheadAttention(embed_dim=1, num_heads=1)
        self.lstm = nn.LSTM(self.input_size, 
                            hidden_size, 
                            num_layers, 
                            batch_first=True, 
                            bidirectional=self.bidirectional,
                            dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x is of shape (batch_size, seq_len, input_size)
        # hidden is of shape (num_layers * num_directions, batch_size, hidden_size)
        # cell is of shape (num_layers * num_directions, batch_size, hidden_size)
        batch_size = x.shape[0]
        seq_len = x.shape[1] 
        x = x.reshape(batch_size, seq_len, -1)
        if self.attention:
            x = self.attention_layer(x, x, x)[0]
            # print ("attention shape: ", x.shape) # (batch_size, seq_len, input_size) input_size = 1
        
        # print ("input shape: ", x.shape) # (batch_size, seq_len, input_size) input_size = 1
        out, (ht, ct) = self.lstm(x) # ht is of shape (num_layers * num_directions, batch_size, hidden_size)
        
        # last hidden state bi-directional LSTM average
        # average hidden states of all directions
        if self.bidirectional:
            out = (ht[-2,:,:] + ht[-1,:,:])/2 
            out = out.reshape(-1, self.hidden_size)
        else:
            out = ht[-1,:,:]

        # print ("out shape: ", out.shape)
        # out is of shape (batch_size, seq_len, num_directions * hidden_size)
        out = self.fc(out)
        # out is of shape (batch_size, num_classes)
        return out

class ModelMetaData:
    def __init__(self, 
                 hidden_size, 
                 num_classes=5, 
                 num_layers=2, 
                 attention=True, 
                 dropout=0.5, 
                 
                 bidirectional=True):
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.attention = attention
        self.dropout = dropout
        self.bidirectional = bidirectional

    def __str__(self):
        return dict(hidden_size=self.hidden_size,
                    num_classes=self.num_classes,
                    num_layers=self.num_layers,
                    attention=self.attention,
                    dropout=self.dropout,
                    bidirectional=self.bidirectional).__str__()
    
    
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

    model = LSTMmodel(hidden_size=128, num_classes=n_classes, num_layers=2, attention=True)

    for x, y in dataloader:
        out = model(x)
        print(out.shape)
        break
