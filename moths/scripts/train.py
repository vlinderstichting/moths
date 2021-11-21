import logging
from dataclasses import MISSING, dataclass

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from pytorch_lightning import seed_everything

from moths.config import prepare_config
from moths.data_module import DataConfig, DataModule
from moths.lit_module import LitConfig
from moths.model import ModelConfig


@dataclass
class Config(DictConfig):
    data: DataConfig = MISSING
    lit: LitConfig = MISSING
    model: ModelConfig = MISSING

    seed: int
    debug: bool
    test: bool


cs = ConfigStore.instance()
cs.store(name="code_config", node=Config)

log = logging.getLogger(__name__)


def train(config: Config) -> None:
    prepare_config(config)
    seed_everything(config.seed, workers=True)

    data_module = DataModule(config.data)

    log.info(f"Loaded {data_module}!")


@hydra.main(config_path="../config", config_name="config")
def train_wrap(config: Config) -> None:
    train(config)


if __name__ == "__main__":
    train_wrap()
