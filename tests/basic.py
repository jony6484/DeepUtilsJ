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



class ConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, activation=nn.ReLU()):
        super().__init__()
        self.layers = [
            nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(num_features=out_ch),
            ]
        if activation is not None:
            self.layers.append(activation)
        self.layers = nn.Sequential(*self.layers)
    
    def forward(self, X):
        return self.layers(X)



class DensCnnEncoder1D(nn.Module):
    def __init__(self, in_channels, channel_list, output_size, kernel_size):
        super().__init__()
        input_ii = in_channels
        self.conv_layers = []
        for output_ch_ii in channel_list:
            self.conv_layers.append(ConvBlock1D(input_ii, output_ch_ii, kernel_size=kernel_size))
            self.conv_layers.append(nn.MaxPool1d(kernel_size=2))
            input_ii = output_ch_ii
        self.conv_layers.append(ConvBlock1D(output_ch_ii, output_ch_ii, kernel_size=kernel_size))
        self.conv_layers.append(nn.AdaptiveAvgPool1d(output_size=output_size))
        self.conv_layers = nn.Sequential(*self.conv_layers)

    def forward(self, X):
        Z = self.conv_layers(X)        
        return Z


class ConstCnnEncoder1D(nn.Module):
    def __init__(self, in_channels, const_channels, n_const_layers, kernel_size):
        super().__init__()
        self.conv_layers = []
        self.conv_layers.append(ConvBlock1D(in_channels, const_channels, kernel_size=kernel_size))
        for ii in range(n_const_layers):
            self.conv_layers.append(ConvBlock1D(const_channels, const_channels, kernel_size=kernel_size))
        self.conv_layers = nn.Sequential(*self.conv_layers)
    def forward(self, X):
        Z = self.conv_layers(X)        
        return Z


class ResCnnBlock1D(nn.Module):
    def __init__(self, n_channels, kernel_size):
        super().__init__()
        self.res_layers = [ConvBlock1D(n_channels, n_channels, kernel_size=kernel_size, activation=None),
                           ConvBlock1D(n_channels, n_channels, kernel_size=kernel_size)]
        self.res_layers = nn.Sequential(*self.res_layers)
        self.activation = nn.ReLU()
    def forward(self, X):
        X = self.activation(self.res_layers(X) + X)
        return X


class ResCnnEncoder1D(nn.Module):
    def __init__(self, in_channels, const_channels, n_res_blocks, n_downsample_layers, output_size, kernel_size):
        super().__init__()
        self.layers = []
        self.layers.append(ConvBlock1D(in_channels, const_channels, kernel_size=kernel_size))
        for ii in range(n_res_blocks):
            self.layers.append(ResCnnBlock1D(const_channels, kernel_size=kernel_size))
        self.layers.append(DensCnnEncoder1D(in_channels=const_channels, channel_list=[const_channels]*n_downsample_layers, output_size=output_size, kernel_size=kernel_size))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, X):
        X = self.layers(X)
        return X


X = torch.randn(20, 1, 100)
conv = DensCnnEncoder1D(1, [2,4,8,16], 2, 3)
model = ResCnnEncoder1D(in_channels=1, const_channels=8, n_res_blocks=4, n_downsample_layers=3, output_size=2, kernel_size=3)
conv(X)
cnn = CnnEncoder(in_features=4, in_len=20, output_size=3, n_blocks=2)
print(cnn(X).shape)
# some_func(['c'])
print('done')