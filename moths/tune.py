import logging

from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.tuner.lr_finder import _LRFinder
from pytorch_lightning.tuner.tuning import Tuner

from moths.config import update_tuned_parameters
from moths.data_module import DataModule
from moths.lit_module import LitModule

log = logging.getLogger("MOTHS")


def tune(
    config: DictConfig, trainer: Trainer, lit_module: LitModule, data_module: DataModule
) -> None:
    # setup
    tuner = Tuner(trainer)
    lit_module._unfreeze_backbone(config.lit.unfreeze_backbone_percentage)

    # batch size
    # tuner.scale_batch_size(lit_module, datamodule=data_module, mode="power")
    # log.info(
    #     f"tuner set the batch size from {config.data.batch_size} to {data_module.batch_size}"
    # )

    # lr
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

    # todo: save the image plot to wandb

    # teardown
    lit_module._freeze_backbone()
    update_tuned_parameters(config, lit_module.lr, data_module.batch_size)
