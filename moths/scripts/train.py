import logging
import os
from dataclasses import dataclass

import hydra
import torch
import wandb
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from moths.config import prepare_config
from moths.data_module import DataConfig, DataModule
from moths.lit_module import LitConfig, LitModule
from moths.logging import log_hyperparameters
from moths.model import Model, ModelConfig
from moths.trainer import TrainerConfig, get_trainer
from moths.tune import tune

log = logging.getLogger("MOTHS")

CONFIG_NAME = os.getenv("MOTHS_CONFIG", "default")


@dataclass
class Config(DictConfig):
    data: DataConfig
    model: ModelConfig
    lit: LitConfig
    trainer: TrainerConfig

    seed: int
    test: bool
    tune: bool


cs = ConfigStore.instance()
cs.store(name="code_config", node=Config)


@hydra.main(config_path="../../config", config_name=CONFIG_NAME)
def train(config: Config) -> None:
    prepare_config(config)
    seed_everything(config.seed, workers=True)

    torch.backends.cudnn.benchmark = True

    data_module = DataModule(config.data)
    model = Model(config.model, data_module.label_hierarchy)
    lit_module = LitModule(config.lit, model, data_module.label_hierarchy)
    trainer = get_trainer(config.trainer)

    log.info("instantiated required objects from config")

    if config.tune:
        tune(config, trainer, lit_module, data_module)

    lit_module.setup()
    log_hyperparameters(
        config=config,
        lit_module=lit_module,
        data_module=data_module,
        trainer=trainer,
    )

    # todo: catch keyboard interrupt
    trainer.fit(lit_module, datamodule=data_module)

    if config.test:
        trainer.test(lit_module, ckpt_path="best", datamodule=data_module)

    if any(isinstance(l, WandbLogger) for l in trainer.logger):
        wandb.finish()

    for callback in trainer.callbacks:
        if isinstance(callback, ModelCheckpoint):
            return callback.best_model_score.detach().cpu().numpy().item()


if __name__ == "__main__":
    train()
