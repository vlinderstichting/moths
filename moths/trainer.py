from dataclasses import dataclass
from typing import Any, Dict, List

from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import LightningLoggerBase


@dataclass
class TrainerConfig(DictConfig):
    instance: Any
    callbacks: Dict[str, Any]
    loggers: Dict[str, Any]


def get_trainer(config: TrainerConfig) -> Trainer:
    callbacks: List[Callback] = [instantiate(c) for c in config.callbacks.values()]
    loggers: List[LightningLoggerBase] = [
        instantiate(c) for c in config.loggers.values()
    ]
    trainer: Trainer = instantiate(
        config.instance, callbacks=callbacks, logger=loggers, _convert_="partial"
    )
    return trainer
