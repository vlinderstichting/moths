import uuid
from pathlib import Path
from typing import List
from uuid import UUID

import numpy as np
from PIL.Image import Image, fromarray
from torch import Tensor

from moths.label_hierarchy import LabelHierarchy


def save_prediction(
    x: Tensor,
    y: Tensor,
    y_hat: List[Tensor],
    label_hierarchy: LabelHierarchy,
    path: Path,
) -> None:
    """

    Args:
        x:
        y:
        y_hat:
        label_hierarchy:
        path:
    """
    x = x.detach().cpu().numpy()
    y_hat = [int(pred.detach().cpu().numpy()) for pred in y_hat]
    species = label_hierarchy.classes[y_hat[0]]

    image_path = path / species / f"{uuid.uuid4().hex}.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)

    # todo: this needs to be not hardcoded
    denormalize_variance_factor = np.array([46.18, 46.70, 48.89]).reshape((3, 1, 1))
    denormalize_mean_factor = np.array([136.24, 133.32, 116.16]).reshape((3, 1, 1))

    x_image = (x * denormalize_variance_factor + denormalize_mean_factor) * 255
    x_image_int = x_image.round().astype(np.uint8)

    x_image_int = np.swapaxes(x_image_int, 0, 2)

    image: Image = fromarray(x_image_int, "RGB")
    image.save(str(image_path))
