import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

log = logging.getLogger(__name__)

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


def from_file(path: Path, classes: Set[str], trim: Set[str]) -> LabelHierarchy:
    with path.open("r") as f:
        reader = csv.reader(f, delimiter=";", quotechar='"')
        _ = next(reader)
        rows = list(reader)

    # hierarchy of classes we know there are going to be filled
    hierarchy_map = {
        klass: (group, family, genus)
        for klass, _, group, family, genus in rows
        if klass in classes and klass not in trim
    }

    log.info(f"Cannot find {len(classes - trim) - len(hierarchy_map)} classes in family data.")

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
    name_map: Dict[str, Tuple[str, str, str]] = {OTHER_NAME: (OTHER_NAME, OTHER_NAME, OTHER_NAME)}

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


#
# def hierarchy_from_path(path: Path, classes: Set[str], trim: Set[str]) -> LabelHierarchy:
#     """
#     classes from trim and classes that are not in the hierarchy from path are put in other
#
#     Args:
#         path:
#         trim:
#
#     Returns:
#
#     """
#     class_index_to_name = {}
#     group_index_to_name = {}
#     family_index_to_name = {}
#     genus_index_to_name = {}
#
#     class_name_to_index = {}
#     group_name_to_index = {}
#     family_name_to_index = {}
#     genus_name_to_index = {}
#
#     hierarchy_index_map = {}
#     hierarchy_name_map = {}
#
#     with path.open("r") as f:
#         lines = [line.strip().split(",") for line in f.readlines()]
#
#     # assumes the file is not corrupt
#     for class_i, klass, group_i, group, family_i, family, genus_i, genus in lines:
#         class_i = int(class_i) + 1
#         group_i = int(group_i) + 1
#         family_i = int(family_i) + 1
#         genus_i = int(genus_i) + 1
#
#         class_index_to_name[class_i] = klass
#         group_index_to_name[group_i] = group
#         family_index_to_name[family_i] = family
#         genus_index_to_name[genus_i] = genus
#
#         class_name_to_index[klass] = class_i
#         group_name_to_index[group] = group_i
#         family_name_to_index[family] = family_i
#         genus_name_to_index[genus] = genus_i
#
#         hierarchy_index_map[class_i] = (group_i, family_i, genus_i)
#         hierarchy_name_map[klass] = (group, family, genus)
#
#     classes = [class_index_to_name[i] for i in range(len(class_index_to_name))]
#     groups = [group_index_to_name[i] for i in range(len(group_index_to_name))]
#     families = [family_index_to_name[i] for i in range(len(family_index_to_name))]
#     genuses = [genus_index_to_name[i] for i in range(len(genus_index_to_name))]
#
#     return LabelHierarchy(
#         classes=classes,
#         groups=groups,
#         families=families,
#         genuses=genuses,
#         class_map=class_name_to_index,
#         group_map=group_name_to_index,
#         family_map=family_name_to_index,
#         genus_map=genus_name_to_index,
#         index_map=hierarchy_index_map,
#         name_map=hierarchy_name_map,
#     )
#
#
#
# def log_hierarchy_artifact(path: Path, label_hierarchy: LabelHierarchy) -> None:
#     """
#     classes from trim and classes that are not in the hierarchy from path are put in other
#
#     Args:
#         path:
#         trim:
#
#     Returns:
#
#     """
#     class_index_to_name = {}
#     group_index_to_name = {}
#     family_index_to_name = {}
#     genus_index_to_name = {}
#
#     class_name_to_index = {}
#     group_name_to_index = {}
#     family_name_to_index = {}
#     genus_name_to_index = {}
#
#     hierarchy_index_map = {}
#     hierarchy_name_map = {}
#
#     with path.open("r") as f:
#         lines = [line.strip().split(",") for line in f.readlines()]
#
#     # assumes the file is not corrupt
#     for class_i, klass, group_i, group, family_i, family, genus_i, genus in lines:
#         class_i = int(class_i) + 1
#         group_i = int(group_i) + 1
#         family_i = int(family_i) + 1
#         genus_i = int(genus_i) + 1
#
#         class_index_to_name[class_i] = klass
#         group_index_to_name[group_i] = group
#         family_index_to_name[family_i] = family
#         genus_index_to_name[genus_i] = genus
#
#         class_name_to_index[klass] = class_i
#         group_name_to_index[group] = group_i
#         family_name_to_index[family] = family_i
#         genus_name_to_index[genus] = genus_i
#
#         hierarchy_index_map[class_i] = (group_i, family_i, genus_i)
#         hierarchy_name_map[klass] = (group, family, genus)
#
#     for klass in classes:
#         if klass in
#
#     classes = [class_index_to_name[i] for i in range(len(class_index_to_name))]
#     groups = [group_index_to_name[i] for i in range(len(group_index_to_name))]
#     families = [family_index_to_name[i] for i in range(len(family_index_to_name))]
#     genuses = [genus_index_to_name[i] for i in range(len(genus_index_to_name))]
#
#     return LabelHierarchy(
#         classes=classes,
#         groups=groups,
#         families=families,
#         genuses=genuses,
#         class_map=class_name_to_index,
#         group_map=group_name_to_index,
#         family_map=family_name_to_index,
#         genus_map=genus_name_to_index,
#         index_map=hierarchy_index_map,
#         name_map=hierarchy_name_map,
#     )
