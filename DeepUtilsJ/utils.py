from pathlib import Path
import importlib.util
import sys

def inverse_dict(d: dict):
    return {v: k for k, v in d.items()}


def validate_dir(new_dir):
    new_dir = Path(new_dir)
    if not new_dir.is_dir():
        Path.mkdir(new_dir)
    return new_dir


def module_file_loader(module_path, module_name="temp_module", import_module=True):
    """
    A Function to load a file as a module
    module_path :: the file location
    module_name :: the local module name for the import, if None, takes the name from the file
    """
    if module_name is None:
        module_name = Path(module_path).name
    # Load the module from file
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if import_module:
        sys.modules[module_name] = module_name
    return module
