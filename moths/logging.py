from copy import deepcopy

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import rank_zero_only

from moths.data_module import DataModule
from moths.lit_module import LitModule


def replace_list_with_dict(d: dict) -> None:
    for k, v in d.items():
        if isinstance(v, dict):
            replace_list_with_dict(v)
        elif isinstance(v, list):
            d[k] = {i: v[i] for i in range(len(v))}
            replace_list_with_dict(d[k])


def target_dict_to_string(d: dict):
    klass = f"{d['_target_'].split('.')[-1]}"
    del d["_target_"]

    if len(d) <= 4:
        params = [f"{str(k)}={str(v)}" for k, v in d.items()]
        return f"{klass}({', '.join(params)})"
    else:
        return {f"{klass}.{str(k)}": str(v) for k, v in d.items()}


def replace_target_with_string(d: dict) -> None:
    for k, v in d.items():
        if isinstance(v, dict):
            if "_target_" in v:
                d[k] = target_dict_to_string(v)
            else:
                replace_target_with_string(v)
        elif isinstance(v, list):
            for i, e in enumerate(v):
                if isinstance(e, dict) and "_target_" in e:
                    v[i] = target_dict_to_string(e)
                elif isinstance(e, dict):
                    replace_target_with_string(e)


@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    lit_module: LitModule,
    data_module: DataModule,
    trainer: Trainer,
) -> None:
    hp = deepcopy(OmegaConf.to_container(config, resolve=True))
    replace_target_with_string(hp)
    replace_list_with_dict(hp)

    hp["data"]["num_classes"] = len(data_module.label_hierarchy.classes)
    hp["data"]["num_train_samples"] = len(data_module.train_dataset)
    hp["data"]["num_val_samples"] = len(data_module.val_dataset)
    hp["data"]["num_test_samples"] = len(data_module.test_dataset)
    hp["model"]["num_parameters"] = sum(p.numel() for p in lit_module.parameters())

    # it says it cant, but it works fine with a dict
    trainer.logger.log_hyperparams(hp)  # type: ignore

    # hacky trick found on the internet
    def null(*args, **kwargs):
        pass

    trainer.logger.log_hyperparams = null
