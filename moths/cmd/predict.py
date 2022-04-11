import logging
import pickle
import shutil
from dataclasses import dataclass
from typing import cast

import hydra
import torch
from hydra.core.config_store import ConfigStore
from pytorch_lightning import seed_everything

from moths.cmd.train import CONFIG_NAME, Config
from moths.config import resolve_config_path
from moths.data_module import DataModule
from moths.label_hierarchy import LabelHierarchy
from moths.lit_module import LitModule
from moths.model import Model
from moths.trainer import get_trainer

log = logging.getLogger("MOTHS")


@dataclass
class PredictConfig(Config):
    # where the artifacts are stored to do inference with
    # usually the output of a hydra run folder (ie, `multirun/date/time/X/`)
    training_output_path: str


cs = ConfigStore.instance()
cs.store(name="code_config", node=PredictConfig)


@hydra.main(config_path="../../config", config_name=CONFIG_NAME)
def predict(config: PredictConfig) -> None:
    seed_everything(config.seed, workers=True)
    torch.backends.cudnn.benchmark = True

    artifact_path = resolve_config_path(config.training_output_path)
    label_artifact_path = artifact_path / "label_hierarchy.pkl"
    ckpt_path = artifact_path / "best.ckpt"

    with label_artifact_path.open("rb") as f:
        label_hierarchy: LabelHierarchy = pickle.load(f)

    data_module = DataModule(config.data)
    data_module.setup()
    model = Model(config.model, label_hierarchy)
    lit_module = LitModule.load_from_checkpoint(
        checkpoint_path=str(ckpt_path),
        config=config.lit,
        model=model,
        label_hierarchy=label_hierarchy,
    )

    trainer = get_trainer(config.trainer)
    trainer.predict(model=lit_module, dataloaders=data_module.val_dataloader())

    # will raise in the predict of lit module if it is None
    predict_output_path = cast(str, config.lit.prediction_output_path)

    predict_path = resolve_config_path(predict_output_path)
    label_hierarchy_dst_path = predict_path / "label_hierarchy.pkl"
    ckpt_dst_path = predict_path / "best.ckpt"

    # by copying these files the predict path folder becomes a complete artifact that can be used for other purposes
    shutil.copy(str(label_artifact_path), str(label_hierarchy_dst_path))
    shutil.copy(str(ckpt_path), str(ckpt_dst_path))


if __name__ == "__main__":
    predict()
