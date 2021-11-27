from pathlib import Path
from typing import Tuple

import numpy as np
import typer
from joblib import Parallel, delayed
from PIL import Image
from tqdm import tqdm

norm_values_app = typer.Typer()


@norm_values_app.command()
def calculate_normalization_values(data_path: Path) -> None:
    def _calc(file_path: Path) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        img = np.array(Image.open(file_path).convert("RGB"))
        return img.mean(axis=(0, 1)), img.var(axis=(0, 1))

    paths = [p for c in data_path.iterdir() for p in c.iterdir()]
    stats = Parallel(n_jobs=24)(delayed(_calc)(p) for p in tqdm(paths))
    stats = np.array(stats).mean(axis=0)

    print("mean:", stats[0].round(2))
    print("std: ", np.sqrt(stats[1]).round(2))


if __name__ == "__main__":
    norm_values_app()
