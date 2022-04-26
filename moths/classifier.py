import pickle
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, cast

import torch

from moths.label_hierarchy import LabelHierarchy
from moths.model import Model, ModelConfig


@dataclass
class ModelConfigStub:
    zoo_name: str
    pretrained: bool


def strip_key(key):
    return key[6:]  # remove "model." from the start of the key


def load_model(path: Path, zoo_name: str) -> Tuple[Model, LabelHierarchy]:
    with (path / "label_hierarchy.pkl").open("rb") as f:
        label_hierarchy: LabelHierarchy = pickle.load(f)

    config = cast(ModelConfig, ModelConfigStub(zoo_name=zoo_name, pretrained=True))

    ckpt = torch.load(str(path / "best.ckpt"), map_location=torch.device("cpu"))
    ckpt_new = OrderedDict(
        [(strip_key(key), value) for key, value in ckpt["state_dict"].items()]
    )

    model = Model(config, label_hierarchy)
    model.load_state_dict(ckpt_new)

    model.eval()

    return model, label_hierarchy
