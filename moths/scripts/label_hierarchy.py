import csv
from pathlib import Path
from typing import List

import typer

LABEL_HIERARCHY_FILE_NAME = "label-hierarchy.csv"

label_hierarchy_app = typer.Typer()


@label_hierarchy_app.command()
def write_label_hierarchy(
    out_path: Path, family_path: Path, data_paths: List[Path]
) -> None:
    """Create a label hierarchy file based on the family.csv and multiple data paths.

    For more information see the README.md.

    Args:
        out_path: the folder to write the result to
        family_path: path to csv file with label hierarchy information
        data_paths: input data folders with structure <path>/<class>/<image>
    """
    classes = {
        class_path.name
        for data_path in data_paths
        for class_path in data_path.iterdir()
        if class_path.is_dir()
    }

    with family_path.open("r") as f:
        reader = csv.reader(f, delimiter=";", quotechar='"')
        _ = next(reader)
        rows = list(reader)

    hierarchy_map = {
        klass: [group, family, genus] for klass, _, group, family, genus in rows
    }
    hierarchy_map = {k: t for k, t in hierarchy_map.items() if k in classes}

    # Note: this ignores all classes not found in the family.csv
    classes = sorted([c for c in classes if c in hierarchy_map])

    groups = sorted(list({g for g, _, _ in hierarchy_map.values()}))
    group_map = {g: i for i, g in enumerate(groups)}

    families = sorted(list({f for _, f, _ in hierarchy_map.values()}))
    family_map = {f: i for i, f in enumerate(families)}

    genuses = sorted(list({g for _, _, g in hierarchy_map.values()}))
    genus_map = {g: i for i, g in enumerate(genuses)}

    lines_out = []
    for klass_i, klass in enumerate(classes):
        group, family, genus = hierarchy_map[klass]
        group_i = group_map[group]
        family_i = family_map[family]
        genus_i = genus_map[genus]

        lines_out.append(
            f"{klass_i},{klass},"
            f"{group_i},{group},"
            f"{family_i},{family},"
            f"{genus_i},{genus}"
            f"\n"
        )

    out_path = out_path / LABEL_HIERARCHY_FILE_NAME
    with out_path.open("w") as f:
        f.writelines(lines_out)

    print(f"Written '{len(classes)}' classes to '{str(out_path)}'")


if __name__ == "__main__":
    label_hierarchy_app()
