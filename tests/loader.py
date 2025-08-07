import torch
from pathlib import Path
import importlib.util
import sys
import pickle
import torch


class A1:
    def __init__(self, a: int):
        self.a = a

    def __call__(self, x: int):
        return self.a + x

# def test_model_loader():
#     # Example usage
# base_path = Path("C:/Users/jonaf/Documents/projects/models/temp")

# model_loader = ModelLoader(base_path)
# model_loader.load_model()
# model_loader.load_weights()

