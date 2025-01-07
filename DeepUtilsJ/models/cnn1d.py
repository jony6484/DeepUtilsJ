import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import DeepUtilsJ.models.model_utils as model_utils


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


class DenseCnnEncoder1D(nn.Module):
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
    def __init__(self, in_channels, const_channels, n_res_blocks, n_downsample_blocks, output_size, kernel_size):
        super().__init__()
        self.layers = []
        self.layers.append(ConvBlock1D(in_channels, const_channels, kernel_size=kernel_size))
        for ii in range(n_res_blocks):
            self.layers.append(ResCnnBlock1D(const_channels, kernel_size=kernel_size))
        self.layers.append(DenseCnnEncoder1D(in_channels=const_channels, channel_list=[const_channels]*n_downsample_blocks, output_size=output_size, kernel_size=kernel_size))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, X):
        X = self.layers(X)
        return X


"""
X = torch.randn(20, 1, 100)
model = ResCnnEncoder1D(in_channels=1, const_channels=8, n_res_blocks=4, n_downsample_layers=3, output_size=2, kernel_size=3)
"""