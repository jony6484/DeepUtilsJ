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