import torch.nn as nn
from collections import OrderedDict


def add_first_layer(model, new_layer, new_layer_name="new_layer"):
    new_layers = OrderedDict()
    new_layers[new_layer_name] = new_layer

    # Add the existing layers to the OrderedDict
    for name, layer in model.named_children():
        new_layers[name] = layer
    model = nn.Sequential(new_layers)
    return model


class ListToDict(nn.Module):
    def __init__(self, keys):
        """       
        Args:
            keys (list of str): List of keys for the output dictionary.
        """
        super(ListToDict, self).__init__()
        self.keys = keys
        
    def forward(self, *tensor_list):
        """
        Converts a list of tensors into a dictionary.
        
        Args:
            tensor_list (list of torch.Tensor): A list of tensors.
            
        Returns:
            dict: A dictionary with keys and corresponding tensors.
        """
        if len(tensor_list) != len(self.keys):
            raise ValueError(f"Number of keys ({len(self.keys)}) must match the number of tensors ({len(tensor_list)}).")
        return {key: tensor for key, tensor in zip(self.keys, tensor_list)}
