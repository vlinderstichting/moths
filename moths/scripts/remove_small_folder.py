from pathlib import Path

import typer
import shutil

remove_small_folder_app = typer.Typer()


@remove_small_folder_app.command()
def remove(path: Path, minimum: int) -> None:
    # 10 is needed to pass
    for class_path in path.iterdir():
        num_files = len(list(class_path.iterdir()))
        if num_files < minimum:
            shutil.rmtree(class_path)
            print(f"removed {str(class_path)} with {num_files} files")


if __name__ == "__main__":
    remove_small_folder_app()
