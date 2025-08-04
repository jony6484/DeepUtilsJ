from pathlib import Path
import importlib.util
import sys
import torch
import yaml

def inverse_dict(d: dict):
    return {v: k for k, v in d.items()}


def validate_dir(new_dir):
    new_dir = Path(new_dir)
    if not new_dir.is_dir():
        Path.mkdir(new_dir)
    return new_dir


class ModelLoader:
    # could be used with trained model of DeppUtilsJ version 1.0.0 and later
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.scripts_path = base_path / "scripts"
        self.load_model_meta()

    def load_model_meta(self):
        with (self.scripts_path / "metadata.yaml").open('r') as file:
            self.metadata = yaml.safe_load(file)

    def load_model(self):
        sys.path.insert(0, str(self.scripts_path))
        module_name = self.metadata['module_name']
        importlib.import_module(module_name)
        self.model = torch.load(self.base_path / "model.pt", weights_only=False)
        sys.path.remove(str(self.scripts_path))

    def load_weights(self, last_epoch=False):
        checkpoint_file = "checkpoint.pt"
        if last_epoch:
            checkpoint_file = "last_epoch.pt"
        checkpoint = torch.load(self.base_path / checkpoint_file, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state'])
        print(f"Model weights loaded from {self.base_path / checkpoint_file}")
