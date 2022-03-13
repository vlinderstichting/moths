import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import hydra
import torch
from hydra.core.config_store import ConfigStore
from pytorch_lightning import seed_everything

from moths.config import resolve_config_path
from moths.data_module import DataModule
from moths.label_hierarchy import LabelHierarchy
from moths.lit_module import LitModule
from moths.model import Model
from moths.scripts.train import CONFIG_NAME, Config
from moths.trainer import get_trainer

log = logging.getLogger("MOTHS")


@dataclass
class PredictConfig(Config):
    artifact_path: str


cs = ConfigStore.instance()
cs.store(name="code_config", node=PredictConfig)


@hydra.main(config_path="../../config", config_name=CONFIG_NAME)
def predict(config: PredictConfig) -> None:
    seed_everything(config.seed, workers=True)
    torch.backends.cudnn.benchmark = True

    artifact_path = resolve_config_path(config.artifact_path)
    label_artifact_path = artifact_path / "label_hierarchy.pkl"
    ckpt_path = artifact_path / "weights.ckpt"

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
    trainer.predict(model=lit_module, dataloaders=data_module.test_dataloader())


if __name__ == "__main__":
    predict()
