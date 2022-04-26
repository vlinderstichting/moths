import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
import typer
from PIL import Image
from torch import nn
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

from moths.classifier import load_model
from moths.label_hierarchy import LabelHierarchy

inference_app = typer.Typer()


@dataclass
class InferenceSampleOutput:
    path: Path
    klass: str
    error_rate: float


def infer_image(
    model: nn.Module, label_hierarchy: LabelHierarchy, path: Path
) -> Tuple[str, float]:

    image = Image.open(str(path)).convert("RGB")

    tfs = Compose(
        [
            Resize(224),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=[136.24, 133.32, 116.16], std=[46.18, 46.70, 48.89]),
        ]
    )

    image = tfs(image)
    image = torch.unsqueeze(image, 0)

    if torch.cuda.is_available():
        image.to("cuda")

    # get first item from batch, get first item from tuple (the class)
    predictions = model.forward(image)[0][0].detach().cpu()

    prediction_argmax = torch.argmax(predictions)
    prediction_class = label_hierarchy.classes[prediction_argmax]
    prediction_score = float(torch.softmax(predictions, 0)[prediction_argmax])

    return prediction_class, prediction_score


@inference_app.command()
def inference(model_path: Path, image_path: Path, result_path: Path) -> None:
    """

    Args:
        model_path: folder that contains the model weights and other artifacts needed to load the model
        image_path: which image to do inference on
        result_path: file to write the results to (must be non-existent)
    """
    model, label_hierarchy = load_model(model_path, "efficientnet_b7")

    if torch.cuda.is_available():
        model.to("cuda")

    klass, score = infer_image(model, label_hierarchy, image_path)

    if result_path.exists():
        raise FileExistsError

    result_path.parent.mkdir(exist_ok=True, parents=True)

    with result_path.open("w") as f:
        json.dump(
            {"path": str(image_path.absolute()), "class": klass, "score": score}, f
        )


if __name__ == "__main__":
    inference_app()
