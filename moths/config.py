from pathlib import Path
from typing import Union

from omegaconf import DictConfig, OmegaConf


def resolve_config_path(path: Union[Path, str]) -> Path:
    """Get the absolute of paths specified in config.

    Because hydra changes the working directory during execution, we need to make sure
    to resolve relative paths to their intended destination.

    If the input path is an absolute path it return the path as is.

    Args:
        path: as specified in the config

    Returns:
        absolute path corrected for the changed working directory
    """
    path = Path(path)
    original_cwd = Path(__file__).parent.parent  # todo: make file location independent
    path = path if path.is_absolute() else original_cwd / path

    return path


def update_tuned_parameters(config: DictConfig, lr: float, batch_size: int) -> None:
    OmegaConf.set_struct(config, False)

    config.lit.optimizer.lr = lr
    config.data.batch_size = batch_size

    OmegaConf.set_struct(config, True)
