from pathlib import Path
import importlib.util
import sys
import torch
import yaml
from inspect import getmodule, getfile, getsourcelines
import shutil
from types import ModuleType
import ast


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


class ModelScriptBackup:
    def __init__(self, backup_dir, project_root):
        self.project_root = Path(project_root).resolve()
        self.backup_dir = Path(backup_dir).resolve()
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def get_all_modules(self, obj):
        modules = set()

        def collect_modules(obj):
            module = getmodule(obj)
            if not isinstance(module, ModuleType):
                return

            try:
                path = Path(getfile(module)).resolve()
                path.relative_to(self.project_root)  # raises ValueError if not under project root
                if not path.is_file():
                    return
            except Exception:
                return

            if path in modules:
                return  # Already visited
            modules.add(path)

            try:
                with open(path, "r", encoding="utf-8") as f:
                    source = f.read()
                tree = ast.parse(source)
            except Exception:
                return

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_name = alias.name
                        try:
                            submod = importlib.import_module(module_name)
                            collect_modules(submod)
                        except Exception:
                            pass

                elif isinstance(node, ast.ImportFrom):
                    base_module = node.module
                    if base_module is None:
                        # handle relative imports like "from . import x"
                        for alias in node.names:
                            rel_mod = "." * node.level + alias.name
                            try:
                                submod = importlib.import_module(rel_mod, module.__name__)
                                collect_modules(submod)
                            except Exception:
                                pass
                    else:
                        for alias in node.names:
                            full_name = f"{base_module}.{alias.name}"
                            try:
                                submod = importlib.import_module(full_name)
                                collect_modules(submod)
                            except Exception:
                                try:
                                    submod = importlib.import_module(base_module)
                                    collect_modules(submod)
                                except Exception:
                                    pass

        collect_modules(obj)
        return list(modules)

    def backup_modules(self, obj):
        module_paths = self.get_all_modules(obj)
        for module_path in module_paths:
            rel_path = module_path.relative_to(self.project_root)
            backup_path = self.backup_dir / rel_path
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(module_path, backup_path)
            print(f"✅ {rel_path}")