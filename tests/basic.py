import path_setup
from DeepUtilsJ.models.autoencoder import LinearCVAE, LinearEncDec
from DeepUtilsJ.losses import VAELoss
import torch
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

model1 = LinearEncDec(5, 3, return_dict=True)
model2 = LinearEncDec(5, 3, return_dict=False)


def some_func(lst):
    outputs = {'a':1, 'b':2}
    c=3
    # l = lst[0]
    for l in lst:
        outputs[l] = locals()[l]
    print(outputs)
    
some_func(['c'])
print('done')