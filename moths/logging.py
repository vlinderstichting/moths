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

    hp["data"] = config.data
    hp["data/num_classes"] = data_module.num_classes
    hp["data/num_train_samples"] = len(data_module.train_dataset)
    hp["data/num_val_samples"] = len(data_module.val_dataset)
    hp["data/num_test_samples"] = len(data_module.test_dataset)

    for name in ["train", "test"]:
        # todo: as one list
        # del hp[f"data/{name}_transforms"]
        for i, cfg in enumerate(config.data[f"{name}_transforms"]):
            # todo: add kwargs

            hp[f"data/{name}_transform_{i}"] = f"{cfg['_target_'].split('.')[-1]}"

    hp["model"] = config.model
    hp["model/num_parameters"] = sum(p.numel() for p in lit_module.parameters())

    # TODO: add everything to here, probably flattened?
    # TODO: how to deal with all these class instances?

    # it says it cant, but it works fine with a dict
    trainer.logger.log_hyperparams(hp)  # type: ignore

    # hacky trick found on the internet
    def empty(*args, **kwargs):
        pass

    trainer.logger.log_hyperparams = empty
