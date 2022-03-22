import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import hydra
import torch
import wandb
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from moths.config import CONFIG_NAME
from moths.data_module import DataConfig, DataModule
from moths.lit_module import LitConfig, LitModule
from moths.logging import log_hyperparameters
from moths.model import Model, ModelConfig
from moths.trainer import TrainerConfig, get_trainer
from moths.tune import tune

log = logging.getLogger("MOTHS")


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
    seed_everything(config.seed, workers=True)

    torch.backends.cudnn.benchmark = True

    data_module = DataModule(config.data)
    data_module.setup()
    model = Model(config.model, data_module.label_hierarchy)
    lit_module = LitModule(config.lit, model, data_module.label_hierarchy)
    trainer = get_trainer(config.trainer)

    log.info("instantiated objects from config")

    if config.tune:
        tune(config, trainer, lit_module, data_module)

    lit_module.setup()
    log_hyperparameters(
        config=config,
        lit_module=lit_module,
        data_module=data_module,
        trainer=trainer,
    )

    try:
        trainer.fit(lit_module, datamodule=data_module)
    except (RuntimeError, KeyboardInterrupt) as e:
        log.warning(e)
        log.info("Abort fitting ... ")
        log.info("Finish up the rest of the train script ... ")

    if config.test:
        trainer.test(lit_module, ckpt_path="best", datamodule=data_module)

    if any(isinstance(l, WandbLogger) for l in trainer.logger):
        wandb.finish()

    for callback in trainer.callbacks:
        if isinstance(callback, ModelCheckpoint):
            current_score = callback.current_score.detach().cpu().numpy().item()
            best_score = callback.best_model_score.detach().cpu().numpy().item()

            best_model_path_src = Path(callback.best_model_path)
            best_model_path_dst = Path(os.getcwd()) / "best.ckpt"

            shutil.copy(str(best_model_path_src), str(best_model_path_dst))

            log.debug(f"Last score: {current_score}")
            log.info(f"Best score: {best_score}")
            log.info(f"Written best weights to: {str(best_model_path_dst.absolute())}")

            return best_score


if __name__ == "__main__":
    train()
