import logging
import os
from dataclasses import dataclass

import hydra
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from pytorch_lightning import seed_everything
from pytorch_lightning.tuner.lr_finder import _LRFinder
from pytorch_lightning.tuner.tuning import Tuner

from moths.config import prepare_config, update_tuned_parameters
from moths.data_module import DataConfig, DataModule
from moths.lit_module import LitConfig, LitModule
from moths.logging import log_hyperparameters
from moths.model import Model, ModelConfig
from moths.trainer import TrainerConfig, get_trainer

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

    # tuning setup
    tuner = Tuner(trainer)
    lit_module._unfreeze_backbone(config.lit.unfreeze_backbone_percentage)

    # tuning
    tuner.scale_batch_size(lit_module, datamodule=data_module, mode="binsearch")
    log.info(
        f"tuner set the batch size from {config.data.batch_size} to {data_module.batch_size}"
    )

    lr_result: _LRFinder = tuner.lr_find(lit_module, datamodule=data_module)
    new_lr = lr_result.suggestion()
    if new_lr is not None:
        lit_module.lr = new_lr
        log.info(
            f"tuner set the learning rate from {config.lit.optimizer.lr} to {lit_module.lr}"
        )
    else:
        log.warning(
            f"could not auto tune lr, leaving the learning rate at {lit_module.lr}"
        )

    # tuning teardown
    lit_module._freeze_backbone()
    update_tuned_parameters(config, lit_module.lr, data_module.batch_size)

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
