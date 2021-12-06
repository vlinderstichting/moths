import logging
from dataclasses import dataclass
from typing import Tuple

from omegaconf import DictConfig
from torch import Tensor, nn
from torchvision import models
from torchvision.models import EfficientNet, RegNet, ResNet

from moths.label_hierarchy import LabelHierarchy

log = logging.getLogger(__name__)


def _get_in_features_and_set_identify(backbone: nn.Module) -> int:
    if isinstance(backbone, EfficientNet):
        in_features = backbone.classifier[1].in_features
        backbone.classifier[1] = nn.Identity()

    elif isinstance(backbone, (ResNet, RegNet)):
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

    else:
        raise NotImplementedError(type(backbone))

    return in_features


def get_net_layers(node, out):
    # unused: might come in handy if .named_parameters turns out to return layers out of order
    children = list(node.children())

    # order is important
    # first do children
    # then do leaves as "out + leaf"

    if len(children) > 0:
        for child in node.children():
            out = get_net_layers(child, out)
        return out
    else:
        return out + [node]


@dataclass
class ModelConfig(DictConfig):
    zoo_name: str
    pretrained: bool


class Model(nn.Module):
    def __init__(self, config: ModelConfig, hierarchy: LabelHierarchy) -> None:
        super(Model, self).__init__()

        model_fn = getattr(models, config.zoo_name)
        self.backbone: nn.Module = model_fn(pretrained=config.pretrained)

        # children = _get_children(self.backbone, [])

        in_features = _get_in_features_and_set_identify(self.backbone)

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
