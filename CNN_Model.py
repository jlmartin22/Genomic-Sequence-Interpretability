import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights(m):
    if isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
        
class ConvNet(nn.Module):
    def __init__(self, seq_len, motif_len, dropout, lay1_sz, lay2_sz, lnlay):
        super(ConvNet, self).__init__()
        self.size = int(np.ceil((((seq_len-motif_len)/2)-motif_len)/2)*lay2_sz)
        self.layer1 = nn.Sequential(
            nn.Conv1d(4, lay1_sz, motif_len, 1),
            nn.ReLU(),
            nn.MaxPool1d(2, 2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(lay1_sz, lay2_sz, motif_len, 1),
            nn.ReLU(),
            nn.MaxPool1d(2, 2))
        self.drop_out = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.size , lnlay)

        self.layer1.apply(init_weights)
        self.layer2.apply(init_weights)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.fill_(0.01)
        
    def forward(self, x):
#         conv = F.conv1d(x, weight)
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        return out