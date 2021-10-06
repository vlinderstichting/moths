import logging
from dataclasses import dataclass

from omegaconf import DictConfig
from torch import nn
from torchvision import models

log = logging.getLogger(__name__)


@dataclass
class ModelConfig(DictConfig):
    zoo_name: str = "resnet50"
    pretrained: bool = True


def get_model(config: ModelConfig, num_classes: int) -> nn.Module:
    if not hasattr(models, config.zoo_name):
        raise ValueError()

    model_fn = getattr(models, config.zoo_name)
    model: nn.Module = model_fn(pretrained=config.pretrained)

    # TODO: different architectures might have different last layers
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model
