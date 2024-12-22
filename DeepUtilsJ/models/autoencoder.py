import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import DeepUtilsJ.models.model_utils as model_utils


class LinearEncDec(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=None, p_drop=0.0, return_dict=False) -> None:
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = []
        sizes_enc = [input_size] + hidden_sizes + [output_size]
        layer_lst = []
        for ii in range(len(sizes_enc) - 2):
            layer_lst += [nn.Linear(in_features=sizes_enc[ii], out_features=sizes_enc[ii+1]), nn.BatchNorm1d(num_features=sizes_enc[ii+1]), nn.Dropout1d(p_drop), nn.ReLU()]
        layer_lst.append(nn.Linear(in_features=sizes_enc[-2], out_features=sizes_enc[-1]))
        if return_dict:
            layer_lst.append(model_utils.ListToDict(keys=['Z']))
        self.layers = nn.Sequential(*layer_lst)

    def forward(self, X):
        Z = self.layers(X)
        return Z


class LinearAE(nn.Module):
    def __init__(self, input_size, bottleneck_dim, enc_hidden_sizes=None, p_drop=0.0, return_dict=False) -> None:
        super().__init__()
        if enc_hidden_sizes is None:
            dec_hidden_sizes = None
        else:
            dec_hidden_sizes = enc_hidden_sizes[::-1]
        self.encoder = LinearEncDec(input_size, bottleneck_dim, enc_hidden_sizes, p_drop=p_drop)
        self.decoder = LinearEncDec(bottleneck_dim, input_size, dec_hidden_sizes, p_drop=p_drop)
        self.return_dict = return_dict
        if return_dict:
            self.dict_output = model_utils.ListToDict(keys=['X_hat'])

    def forward(self, X):
        Z = self.encoder(X)
        X_hat = self.decoder(Z)
        if self.return_dict:
            X_hat = self.dict_output(X_hat)
        return X_hat
    

class LinearVAE(nn.Module):
    def __init__(self, input_size, bottleneck_dim, enc_hidden_sizes=None, p_drop=0.0, return_dict=False) -> None:
        super().__init__()
        if enc_hidden_sizes is None:
            dec_hidden_sizes = None
        else:
            dec_hidden_sizes = enc_hidden_sizes[::-1]
        self.encoder = LinearEncDec(input_size, bottleneck_dim * 2, enc_hidden_sizes, p_drop=p_drop) # mu + var
        self.decoder = LinearEncDec(bottleneck_dim, input_size    , dec_hidden_sizes, p_drop=p_drop)
        self.return_dict = return_dict
        if return_dict:
            self.dict_output = model_utils.ListToDict(keys=['X_hat', 'mu', 'log_var'])

    def forward(self, X):
        X = self.encoder(X)
        mu, log_var = X.chunk(2, dim=1)
        if self.training == True:
            e = torch.randn_like(mu)
            sigma = torch.exp(.5 * log_var)
            Z = sigma * e + mu
        else:
            Z = mu
        X_hat = self.decoder(Z)
        outputs = X_hat, mu, log_var
        if self.return_dict:
            outputs = self.dict_output(outputs)
        return outputs
    

class LinearCVAE(nn.Module):
    def __init__(self, n_classes, input_size, neck_dim, enc_hidden_sizes=None, class_hidden_sizes=None, p_drop=0.0, neck_activation=None, return_dict=False) -> None:
        super().__init__()
        if enc_hidden_sizes is None:
            dec_hidden_sizes = None 
        else:
            dec_hidden_sizes = enc_hidden_sizes[::-1]
        self.n_classes = n_classes
        self.encoder    = LinearEncDec(input_size, neck_dim * 2, enc_hidden_sizes, p_drop=p_drop) # mu + var
        self.decoder    = LinearEncDec(neck_dim, input_size    , dec_hidden_sizes, p_drop=p_drop)
        self.classifier = LinearEncDec(neck_dim, n_classes   , class_hidden_sizes, p_drop=p_drop) 
        if neck_activation is not None:
            self.decoder = model_utils.add_first_layer(self.decoder, neck_activation, new_layer_name="neck_activation")
            self.classifier = model_utils.add_first_layer(self.classifier, neck_activation, new_layer_name="neck_activation")       
        self.return_dict = return_dict
        if return_dict:
            self.dict_output = model_utils.ListToDict(keys=['X_hat', 'mu', 'log_var', 'Y_hat'])
            
    def forward(self, X):
        Z = self.encoder(X)
        mu, log_var = Z.chunk(2, dim=1)
        if self.training == True:
            e = torch.randn_like(mu)
            sigma = torch.exp(.5 * log_var)
            Z = sigma * e + mu
        else:
            Z = mu
        X_hat = self.decoder(Z)        
        Y_hat = self.classifier(Z)
        outputs = X_hat, mu, log_var, Y_hat
        if self.return_dict:
            outputs = self.dict_output(outputs)
        return outputs


def calc_triangle_sizes(input_size, output_size, n_hidden_layers):
    k = (output_size / input_size)**(1/(n_hidden_layers+1))
    sizes_list = [int(input_size*k**n) for n in range(n_hidden_layers+1)] + [output_size]
    return sizes_list