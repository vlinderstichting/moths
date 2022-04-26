from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import typer

from moths.classifier import load_model

retrain_classifier_app = typer.Typer()


@dataclass
class ModelConfig:
    zoo_name: str
    pretrained: bool


def strip_key(key):
    return key[6:]  # remove "model." from the start of the key


@retrain_classifier_app.command()
def retrain_classifier_app(path: Path) -> None:
    model = load_model(path, "efficientnet_b7")

    weights = model.fc_class.weight
    bias = model.fc_class.bias

    species_y = torch.Tensor(np.load(str(path / "arrays" / "0_species_y.npy")))
    features = torch.Tensor(np.load(str(path / "arrays" / "features.npy")))


if __name__ == "__main__":
    retrain_classifier_app()
