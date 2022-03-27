import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

log = logging.getLogger("MOTHS")

LABELS = ["species", "group", "family", "genus"]
OTHER_NAME = "Other"


@dataclass
class LabelHierarchy:
    classes: List[str]
    groups: List[str]
    families: List[str]
    genuses: List[str]

    class_map: Dict[str, int]
    group_map: Dict[str, int]
    family_map: Dict[str, int]
    genus_map: Dict[str, int]

    index_map: Dict[int, Tuple[int, int, int]]
    name_map: Dict[str, Tuple[str, str, str]]


def get_classes_by_label(hierarchy: LabelHierarchy, label: str):
    if label == "species":
        return hierarchy.classes
    if label == "group":
        return hierarchy.groups
    if label == "family":
        return hierarchy.families
    if label == "genus":
        return hierarchy.genuses

    raise ValueError("unknown label")


def label_hierarchy_from_file(
    path: Path, data_path: Path, min_samples: int
) -> LabelHierarchy:
    class_counts = {p.name: len(list(p.iterdir())) for p in data_path.iterdir()}
    classes = set(class_counts.keys())
    classes_to_trim = {c for c, n in class_counts.items() if n < min_samples}
    log.info(
        f"Found {len(classes)} classes. {len(classes_to_trim)} have less than {min_samples} samples."
    )

    with path.open("r") as f:
        reader = csv.reader(f, delimiter=";", quotechar='"')
        _ = next(reader)
        rows = list(reader)

    # hierarchy of classes we know there are going to be filled
    hierarchy_map = {
        klass: (group, family, genus)
        for klass, _, group, family, genus in rows
        if klass in classes and klass not in classes_to_trim
    }

    log.info(
        f"Cannot find {len(classes - classes_to_trim) - len(hierarchy_map)} classes in family data."
    )

    # in: index to name
    # ni: name to index

    class_in = [OTHER_NAME] + sorted(list({c for c in hierarchy_map.keys()}))
    class_ni = {c: i for i, c in enumerate(class_in)}

    group_in = [OTHER_NAME] + sorted(list({g for g, _, _ in hierarchy_map.values()}))
    group_ni = {g: i for i, g in enumerate(group_in)}

    family_in = [OTHER_NAME] + sorted(list({f for _, f, _ in hierarchy_map.values()}))
    family_ni = {f: i for i, f in enumerate(family_in)}

    genus_in = [OTHER_NAME] + sorted(list({g for _, _, g in hierarchy_map.values()}))
    genus_ni = {g: i for i, g in enumerate(genus_in)}

    index_map: Dict[int, Tuple[int, int, int]] = {0: (0, 0, 0)}
    name_map: Dict[str, Tuple[str, str, str]] = {
        OTHER_NAME: (OTHER_NAME, OTHER_NAME, OTHER_NAME)
    }

    for class_n, (group_n, family_n, genus_n) in hierarchy_map.items():
        class_i = class_ni[class_n]
        group_i = group_ni[group_n]
        family_i = family_ni[family_n]
        genus_i = genus_ni[genus_n]

        index_map[class_i] = (group_i, family_i, genus_i)
        name_map[class_n] = (group_n, family_n, genus_n)

    return LabelHierarchy(
        classes=class_in,
        groups=group_in,
        families=family_in,
        genuses=genus_in,
        class_map=class_ni,
        group_map=group_ni,
        family_map=family_ni,
        genus_map=genus_ni,
        index_map=index_map,
        name_map=name_map,
    )
