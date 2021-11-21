from pathlib import Path
from typing import List, Tuple

import typer
from joblib import Parallel, delayed
from PIL import Image
from tqdm import tqdm

from moths.label_hierarchy import hierarchy_from_path

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
            return file_path, False
        else:
            return file_path, True

    path_is_valids = Parallel(n_jobs=10)(
        delayed(_determine_valid)(p) for p in tqdm(paths_to_validate)
    )
    valid_paths = [p for p, is_valid in path_is_valids if is_valid]

    source_out_path = out_path / "image_folder"
    for i, img_path in enumerate(valid_paths):
        class_name = img_path.parent

        symbolic_link_path = (source_out_path / class_name / f"{i:07}").with_suffix(
            img_path.suffix
        )
        symbolic_link_path.symlink_to(img_path)

    print(f"Written {len(valid_paths)} valid symbolic links to {str(source_out_path)}")
    print(f"Skipped {len(paths_to_validate) - len(valid_paths)} images")


if __name__ == "__main__":
    valid_path_app()
