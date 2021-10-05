from pathlib import Path
from typing import Union

import hydra


def resolve_original_path(path: Union[Path, str]) -> Path:
    path = Path(path)
    original_cwd = Path(hydra.utils.get_original_cwd())
    path = path if path.is_absolute() else original_cwd / path

    return path
