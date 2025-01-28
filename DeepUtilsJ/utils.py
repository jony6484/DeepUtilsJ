from pathlib import Path

def inverse_dict(d: dict):
    return {v: k for k, v in d.items()}


def validate_dir(new_dir):
    new_dir = Path(new_dir)
    if not new_dir.is_dir():
        Path.mkdir(new_dir)
    return new_dir
