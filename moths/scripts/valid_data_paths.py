from pathlib import Path
from typing import List

import typer
from PIL import Image
from tqdm import tqdm

VALID_PATH_FILE_NAME = "valid-paths.txt"
INVALID_PATH_FILE_NAME = "invalid-paths.txt"

valid_path_app = typer.Typer()


@valid_path_app.command()
def write_valid_paths(out_path: Path, data_paths: List[Path]) -> None:
    """Creates file with valid and invalid paths for a list of data paths.

    Args:
        out_path: folder to write output to
        data_paths: input data folders with structure <path>/<class>/<image>
    """
    valid_paths = set()
    invalid_paths = set()

    paths_to_validate = []

    for data_path in data_paths:
        for class_path in data_path.iterdir():

            if not class_path.is_dir():
                typer.echo(f"Ignoring {class_path}...")
                continue

            for file_path in class_path.iterdir():
                paths_to_validate.append(file_path.absolute())

    for file_path in tqdm(paths_to_validate):
        try:
            Image.open(file_path).convert("RGB")
        except Exception as e:
            print(f"[{str(file_path)}] {repr(e)}")
            invalid_paths.add(file_path.name)
        else:
            valid_paths.add(file_path.name)

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
