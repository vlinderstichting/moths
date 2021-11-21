from pathlib import Path
from typing import List, Tuple

import typer
from joblib import Parallel, delayed
from PIL import Image
from tqdm import tqdm

from moths.label_hierarchy import hierarchy_from_path

VALID_PATH_FILE_NAME = "valid-paths.txt"
INVALID_PATH_FILE_NAME = "invalid-paths.txt"

valid_path_app = typer.Typer()


@valid_path_app.command()
def write_valid_paths(
    out_path: Path, label_hierarchy_path: Path, data_paths: List[Path]
) -> None:
    """Creates file with valid and invalid paths for a list of data paths.

    Args:
        out_path: folder to write output to
        data_paths: input data folders with structure <path>/<class>/<image>
    """
    paths_to_validate = []

    hierarchy = hierarchy_from_path(label_hierarchy_path)
    classes = set(hierarchy.classes)

    for data_path in data_paths:
        for class_path in data_path.iterdir():

            if not class_path.is_dir() or class_path.name not in classes:
                print(f"Ignoring {class_path}...")
                continue

            for file_path in class_path.iterdir():
                paths_to_validate.append(file_path.absolute())

    def _determine_valid(file_path) -> Tuple[Path, bool]:
        try:
            Image.open(file_path).convert("RGB")
        except Exception as e:
            print(f"[{str(file_path)}] {repr(e)}")
            return file_path.name, False
        else:
            return file_path.name, True

    path_is_valids = Parallel(n_jobs=10)(
        delayed(_determine_valid)(p) for p in tqdm(paths_to_validate)
    )

    valid_paths = [p for p, is_valid in path_is_valids if is_valid]
    invalid_paths = [p for p, is_valid in path_is_valids if not is_valid]

    # for file_path in tqdm(paths_to_validate):
    #     try:
    #         Image.open(file_path).convert("RGB")
    #     except Exception as e:
    #         print(f"[{str(file_path)}] {repr(e)}")
    #         invalid_paths.add(file_path.name)
    #     else:
    #         valid_paths.add(file_path.name)

    valid_paths_path = out_path / VALID_PATH_FILE_NAME
    invalid_paths_path = out_path / INVALID_PATH_FILE_NAME

    with valid_paths_path.open("w") as f:
        f.writelines([f"{str(p)}\n" for p in valid_paths])
    with invalid_paths_path.open("w") as f:
        f.writelines([f"{str(p)}\n" for p in invalid_paths])

    print(f"Written valid paths to {str(valid_paths_path)}")
    print(f"Written invalid paths to {str(invalid_paths_path)}")


if __name__ == "__main__":
    valid_path_app()
