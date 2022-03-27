import uuid
from pathlib import Path
from typing import List

import numpy as np
from PIL.Image import Image, fromarray
from torch import Tensor

from moths.label_hierarchy import LabelHierarchy


def save_prediction(
    x: Tensor,
    y: List[Tensor],
    y_hat: List[Tensor],
    label_hierarchy: LabelHierarchy,
    path: Path,
) -> None:
    """Save an image into a folder with the name of the predicted species.

    Args:
        x: CHW tensor of a single image
        y_hat: predicted class per level,
        label_hierarchy: the used label hierarchy by the model (for the mapping of int to class name)
        path: where to write the images to
    """
    x = x.detach().cpu().numpy()
    y_hat = [int(pred.detach().cpu().numpy()) for pred in y_hat]
    species_pred = label_hierarchy.classes[y_hat[0]]
    species_true = label_hierarchy.classes[y[0]]

    is_correct = species_pred == species_true

    prefix = "correct " if is_correct else "wrong "
    suffix = "" if is_correct else f" ({species_true})"

    image_path = path / species_pred / f"{prefix}{uuid.uuid4().hex}{suffix}.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)

    # todo: this needs to be not hardcoded
    denormalize_variance_factor = np.array([46.18, 46.70, 48.89]).reshape((3, 1, 1))
    denormalize_mean_factor = np.array([136.24, 133.32, 116.16]).reshape((3, 1, 1))

    x_image = (x * denormalize_variance_factor + denormalize_mean_factor) * 255
    x_image_int = x_image.round().astype(np.uint8)

    x_image_int = np.swapaxes(x_image_int, 0, 2)

    image: Image = fromarray(x_image_int, "RGB")
    image.save(str(image_path))
