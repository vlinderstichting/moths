from dataclasses import MISSING, dataclass

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from pytorch_lightning import seed_everything

from moths.config import prepare_config
from moths.data_module import DataConfig, DataModule
from moths.lit_module import LitConfig, LitModule
from moths.logging import log_hyperparameters
from moths.model import ModelConfig, get_model
from moths.trainer import TrainerConfig, get_trainer


@dataclass
class Config(DictConfig):
    data: DataConfig = MISSING
    model: ModelConfig = MISSING
    lit: LitConfig = MISSING
    trainer: TrainerConfig = MISSING

    seed: int = 31415
    debug: bool = False
    test: bool = True


cs = ConfigStore.instance()
cs.store(name="code_config", node=Config)


@hydra.main(config_path="../config", config_name="config")
def train(config: Config) -> None:
    prepare_config(config)
    seed_everything(config.seed, workers=True)

    data_module = DataModule(config.data)
    data_module.prepare_data()  # manually prepare data to set num_classes (to config?)
    data_module.setup()
    model = get_model(config.model, data_module.num_classes)
    lit_module = LitModule(config.lit, model)
    trainer = get_trainer(config.trainer)

    log_hyperparameters(
        config=config,
        lit_module=lit_module,
        data_module=data_module,
        trainer=trainer,
    )

    trainer.fit(lit_module, datamodule=data_module)

    if config.test:
        trainer.test(lit_module, ckpt_path="best", datamodule=data_module)


if __name__ == "__main__":
    train()
