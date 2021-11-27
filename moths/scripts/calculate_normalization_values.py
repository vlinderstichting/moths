from pathlib import Path
from typing import List, Tuple

import typer
from PIL.ImageStat import Stat
from joblib import Parallel, delayed
from PIL import Image
from tqdm import tqdm

import numpy as np

norm_values_app = typer.Typer()

OUTPUT_FOLDER_NAME = "image_folder"


@norm_values_app.command()
def calculate_normalization_values(data_path: Path) -> None:

    def _calc(file_path: Path) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        img = np.array(Image.open(file_path).convert("RGB"))
        return img.mean(axis=(0, 1)), img.std(axis=(0,1))

    paths = [p for c in data_path.iterdir() for p in c.iterdir()]
    stats = Parallel(n_jobs=2)(delayed(_calc)(p) for p in tqdm(paths))

    print(np.array(stats).mean(axis=0))


if __name__ == "__main__":
    norm_values_app()
