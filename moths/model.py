import logging
import time
from dataclasses import dataclass
from typing import Tuple

from omegaconf import DictConfig
from torch import Tensor, nn
from torchvision import models

from moths.label_hierarchy import LabelHierarchy

import torch.nn.functional as F

log = logging.getLogger(__name__)


@dataclass
class ModelConfig(DictConfig):
    zoo_name: str
    pretrained: bool


class DebugBackbone(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        x = F.adaptive_avg_pool2d(x, 1)  # 1 pixel by 3 channels out
        x = x.flatten(1, -1)  # flatten to feature vector size 3
        time.sleep(1)  # to make sure the cpu doesn't go crazy
        return x


class Model(nn.Module):
    def __init__(self, config: ModelConfig, hierarchy: LabelHierarchy) -> None:
        super(Model, self).__init__()
        if config.zoo_name != "debug" and not hasattr(models, config.zoo_name):
            raise ValueError

        if config.zoo_name != "debug":
            model_fn = getattr(models, config.zoo_name)
            self.backbone: nn.Module = model_fn(pretrained=config.pretrained)

            # TODO: different architectures might have different last layer names
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            self.backbone = DebugBackbone()
            in_features = 3

        self.fc_class = nn.Linear(in_features, len(hierarchy.classes))
        self.fc_group = nn.Linear(in_features, len(hierarchy.groups))
        self.fc_family = nn.Linear(in_features, len(hierarchy.families))
        self.fc_genus = nn.Linear(in_features, len(hierarchy.genuses))

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        features = self.backbone(x)

        klass = self.fc_class(features)
        group = self.fc_group(features)
        family = self.fc_family(features)
        genus = self.fc_genus(features)

        return klass, group, family, genus
