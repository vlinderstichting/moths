from pathlib import Path
from typing import List

import typer
from PIL import Image
from tqdm import tqdm


def main(out_path: Path, data_paths: List[Path]):
    """Creates file with valid and invalid paths for a list of data paths.

    Args:
        out_path: Folder where the output files are written.
        data_paths: Input data folders with structure <path>/<class>/<image>
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
            is_valid = True
        except Exception as e:
            typer.echo(f"[{str(file_path)}] {repr(e)}")
            is_valid = False

        if is_valid:
            valid_paths.add(file_path)
        else:
            invalid_paths.add(file_path)

    valid_paths_path = out_path / "valid-paths.txt"
    invalid_paths_path = out_path / "invalid-paths.txt"

    with valid_paths_path.open("w") as f:
        f.writelines([f"{str(p)}\n" for p in valid_paths])
    with invalid_paths_path.open("w") as f:
        f.writelines([f"{str(p)}\n" for p in invalid_paths])

    typer.echo(f"Written valid paths to {str(valid_paths_path)}")
    typer.echo(f"Written invalid paths to {str(invalid_paths_path)}")


if __name__ == "__main__":
    typer.run(main)
