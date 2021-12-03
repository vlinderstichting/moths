import logging
import os
from dataclasses import dataclass

import hydra
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from pytorch_lightning import seed_everything

from moths.config import prepare_config
from moths.data_module import DataConfig, DataModule
from moths.lit_module import LitConfig, LitModule
from moths.logging import log_hyperparameters
from moths.model import Model, ModelConfig
from moths.trainer import TrainerConfig, get_trainer
from moths.tune import tune

log = logging.getLogger(__name__)

CONFIG_NAME = os.getenv("MOTHS_CONF_ENV", "prod")


@dataclass
class Config(DictConfig):
    data: DataConfig
    model: ModelConfig
    lit: LitConfig
    trainer: TrainerConfig

    seed: int = 31415
    debug: bool = False
    test: bool = False


cs = ConfigStore.instance()
cs.store(name="code_config", node=Config)


@hydra.main(config_path="../config", config_name=CONFIG_NAME)
def train(config: Config) -> None:
    prepare_config(config)
    seed_everything(config.seed, workers=True)

    torch.backends.cudnn.benchmark = True

    data_module = DataModule(config.data)
    model = Model(config.model, data_module.label_hierarchy)
    lit_module = LitModule(config.lit, model, data_module.label_hierarchy)
    trainer = get_trainer(config.trainer)

    tune(config, trainer, lit_module, data_module)
    trainer.fit(lit_module, datamodule=data_module)

    # log after data module is instantiated via fit
    log_hyperparameters(
        config=config,
        lit_module=lit_module,
        data_module=data_module,
        trainer=trainer,
    )

    if config.test:
        trainer.test(lit_module, ckpt_path="best", datamodule=data_module)


if __name__ == "__main__":
    train()
