from pathlib import Path
from typing import List

import typer


def main(out_path: Path, data_paths: List[Path]):
    """.

    Args:
        out_path: .
        data_paths: Input data folders with structure <path>/<class>/<image>
    """
    classes = {
        class_path.name
        for data_path in data_paths
        for class_path in data_path.iterdir()
        if class_path.is_dir()
    }
    classes = sorted(list(classes))

    out_path = out_path / "label-map.csv"
    with out_path.open("w") as f:
        f.writelines([f"{i},{str(c)}\n" for i, c in enumerate(classes)])

    typer.echo(f"Written '{len(classes)}' classes to '{str(out_path)}'")


if __name__ == "__main__":
    typer.run(main)
