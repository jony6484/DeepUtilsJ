#%%
import path_setup
# from DeepUtilsJ.models.autoencoder import LinearCVAE, LinearEncDec
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

# model1 = LinearEncDec(5, 3, return_dict=True)
# model2 = LinearEncDec(5, 3, return_dict=False)


def some_func(lst):
    outputs = {'a':1, 'b':2}
    c=3
    # l = lst[0]
    for l in lst:
        outputs[l] = locals()[l]
    print(outputs)
    
#%%
# CNN 1d
N = 100
T = 20
Ft = 4
X = torch.rand(size=(N,Ft, T))

class CnnEncoder(nn.Module):
    def __init__(self, in_features, in_len,  output_size, n_blocks=1, p_drop=0.0, kernel_size=3):
        super().__init__()
        padd = int(kernel_size // 2)
        layer_lst = []
        self.in_features = in_features
        self.in_len = in_len
        self.output_size = output_size
        for ii in range(n_blocks):
            layer_lst += [nn.Conv1d(in_channels=in_features, out_channels=in_features*2 ,kernel_size=kernel_size, padding=padd), nn.BatchNorm1d(in_features*2), nn.MaxPool1d(kernel_size=2), nn.ReLU()]
            in_features *= 2
            in_len = int(in_len // 2)
        layer_lst.append(nn.Flatten(start_dim=1))
        self.layers = nn.Sequential(*layer_lst)
        self.fc = nn.Linear(in_len*in_features, output_size)
        
    def forward(self, X):
        Z = self.layers(X)
        Z = self.fc(Z)
        return Z
    

class CnnDecoder(nn.Module):
    def __init__(self, input_size, out_features, out_len, n_blocks=1, p_drop=0.0, kernel_size=3):
        super().__init__()
        padd = int(kernel_size // 2)
        self.out_features = out_features
        self.out_len = out_len
        for ii in range(n_blocks):
            out_features *= 2
            out_len = int(in_len // 2)
        self.fc = nn.Linear(input_size, out_len*out_features)
        layer_lst = [nn.Unflatten(1, unflattened_size=(-1, out_features, out_len))]
        for ii in range(n_blocks):
            layer_lst += [nn.Conv1d(in_channels=out_features, out_channels=out_features/2 ,kernel_size=kernel_size, padding=padd), nn.BatchNorm1d(out_features*2), nn.Upsample(scale_factor=2), nn.ReLU()]
            in_features /= 2
            in_len = int(in_len // 2)
        layer_lst.append(nn.Flatten(start_dim=1))
        self.layers = nn.Sequential(*layer_lst)
        self.fc = nn.Linear(in_len*in_features, output_size)
        
    def forward(self, X):
        Z = self.layers(X)
        Z = self.fc(Z)
        return Z

cnn = CnnEncoder(in_features=4, in_len=20, output_size=3, n_blocks=2)
print(cnn(X).shape)
# some_func(['c'])
print('done')