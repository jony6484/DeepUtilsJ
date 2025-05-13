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


def module_file_loader(module_path, module_name=None, import_module=True):
    """
    A Function to load a file as a module
    module_path :: the file location
    module_name :: the local module name for the import, if None, takes the name from the file
    """
    if module_name is None:
        module_name = Path(module_path).stem
    # Load the module from file
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if import_module:
        sys.modules[module_name] = module
    return module


def import_attribute_from_file(module_path, module_name, attribute_name):
    """
    A Wraper function that imports a specific attribute into a module
    """
    module = module_file_loader(module_path, module_name=None, import_module=False)
    module_name = module.__name__
    attribute = getattr(module, attribute_name)
    if module_name in sys.modules:
        current_module = sys.modules[module_name]
    else:
        import types
        current_module = types.ModuleType(module_name)
        sys.modules[module_name] = current_module
    # Inject the atribute
    setattr(current_module, attribute_name, attribute)
    print(f"Attribute: {attribute_name} successfully imported into Module: {module_name}")
    print(f'For direct import use: "from {module_name} import {attribute_name}"')
