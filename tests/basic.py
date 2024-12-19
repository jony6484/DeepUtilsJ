import path_setup
from DeepUtilsJ.models.autoencoder import LinearCVAE
import torch.nn as nn
import torch.nn.functional as F


class TempModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(30, 10)
        self.layer2 = nn.ReLU()
    
    def forward(self, X):
        X = self.layer1(X)
        X = self.layer2(X)
        return X

model = LinearCVAE(5, 3000, 2, [700, 100, 30], [20, 10], neck_activation=nn.Tanh())

print('done')