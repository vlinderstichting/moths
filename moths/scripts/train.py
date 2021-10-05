from dataclasses import dataclass

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf

from moths.data_module import DataConfig, DataModule
from moths.lit_module import LitConfig, LitModule
from moths.model import ModelConfig, get_model


@dataclass
class Config:
    data: DataConfig = MISSING
    model: ModelConfig = MISSING
    lit: LitConfig = MISSING
    debug: bool = False


cs = ConfigStore.instance()
cs.store(name="code_config", node=Config)


@hydra.main(config_path="../config", config_name="config")
def train(cfg: Config) -> None:
    print(OmegaConf.to_yaml(cfg))

    data_module = DataModule(cfg.data)
    data_module.prepare_data()
    model = get_model(cfg.model, data_module.num_classes)
    lit_module = LitModule(cfg.lit, model)

    # model = Model(cfg.model)
    # trainer = Trainer(cfg.trainer, model, data_module)
    # trainer.fit()
    # if cfg.test: trainer.test()


if __name__ == "__main__":
    train()
