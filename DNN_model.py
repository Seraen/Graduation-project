import torch
import torch.nn as nn

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()

        self.m = nn.Sequential(
            nn.Linear(34, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(256,256),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(256, 12),
            nn.Tanh())

    def forward(self, x):
        return self.m(x)