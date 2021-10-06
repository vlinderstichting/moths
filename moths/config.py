from pathlib import Path
from typing import Union

import hydra
from omegaconf import DictConfig, OmegaConf


def prepare_config(config: DictConfig) -> None:
    OmegaConf.set_struct(config, False)

    if config.debug:
        config.trainer.instance.fast_dev_run = True
        config.trainer.instance.gpus = 0
        config.data.pin_memory = False
        config.data.num_workers = 0

    OmegaConf.set_struct(config, True)


def resolve_config_path(path: Union[Path, str]) -> Path:
    path = Path(path)
    original_cwd = Path(hydra.utils.get_original_cwd())
    path = path if path.is_absolute() else original_cwd / path

    return path
