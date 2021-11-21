from dataclasses import MISSING, dataclass, field
from typing import Any, List

from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import LightningLoggerBase


@dataclass
class TrainerConfig(DictConfig):
    instance: Any = MISSING
    callbacks: List[Any] = field(default_factory=list)
    loggers: List[Any] = field(default_factory=list)


def get_trainer(config: TrainerConfig) -> Trainer:
    callbacks: List[Callback] = [instantiate(c) for c in config.callbacks]
    loggers: List[LightningLoggerBase] = [instantiate(c) for c in config.loggers]
    trainer: Trainer = instantiate(
        config.instance, callbacks=callbacks, logger=loggers, _convert_="partial"
    )
    return trainer
