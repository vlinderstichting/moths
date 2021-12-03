from pathlib import Path
from typing import Union

from omegaconf import DictConfig, OmegaConf


def prepare_config(config: DictConfig) -> None:
    """Mutate the config by applying logic.

    Currently it does:
    - when in debug mode, set the config to a single thread cpu process

    """
    OmegaConf.set_struct(config, False)

    # todo: maybe loop over all and resolve the path if settings ends with 'path(s)'?

    if config.debug:
        # config.trainer.instance.fast_dev_run = True
        # config.trainer.instance.gpus = 0
        config.data.num_workers = 0
        # remove wandb

    if config.trainer.instance.gpus == 0:
        config.lit.device = "cpu"
        config.data.pin_memory = False

    OmegaConf.set_struct(config, True)


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
    """Mutate the config by applying logic.

    Currently it does:
    - when in debug mode, set the config to a single thread cpu process

    """
    OmegaConf.set_struct(config, False)

    config["tuned"] = dict()

    config["tuned"]["learning_rate"] = lr
    config["tuned"]["batch_size"] = batch_size

    OmegaConf.set_struct(config, True)
