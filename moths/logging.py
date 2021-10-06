from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import rank_zero_only

from moths.data_module import DataModule
from moths.lit_module import LitModule


@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    lit_module: LitModule,
    data_module: DataModule,
    trainer: Trainer,
) -> None:
    hp = {}

    hp["seed"] = config.seed
    hp["data/num_classes"] = data_module.num_classes

    hp["model/params_total"] = sum(p.numel() for p in lit_module.parameters())
    hp["model/params_trainable"] = sum(
        p.numel() for p in lit_module.parameters() if p.requires_grad
    )

    # TODO: add everything to here, probably flattened?
    # TODO: how to deal with all these class instances?

    # it says it cant, but it works fine with a dict
    trainer.logger.log_hyperparams(hp)  # type: ignore

    # hacky trick found on the internet
    def empty(*args, **kwargs):
        pass

    trainer.logger.log_hyperparams = empty
