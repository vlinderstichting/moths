from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


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


def hierarchy_from_path(path: Path) -> LabelHierarchy:
    with path.open("r") as f:
        lines = [line.strip().split(",") for line in f.readlines()]

    class_index_to_name = {}
    group_index_to_name = {}
    family_index_to_name = {}
    genus_index_to_name = {}

    class_name_to_index = {}
    group_name_to_index = {}
    family_name_to_index = {}
    genus_name_to_index = {}

    hierarchy_index_map = {}
    hierarchy_name_map = {}

    # assumes the file is not corrupt
    for class_i, klass, group_i, group, family_i, family, genus_i, genus in lines:
        class_i = int(class_i)
        group_i = int(group_i)
        family_i = int(family_i)
        genus_i = int(genus_i)

        class_index_to_name[class_i] = klass
        group_index_to_name[group_i] = group
        family_index_to_name[family_i] = family
        genus_index_to_name[genus_i] = genus

        class_name_to_index[klass] = class_i
        group_name_to_index[group] = group_i
        family_name_to_index[family] = family_i
        genus_name_to_index[genus] = genus_i

        hierarchy_index_map[class_i] = (group_i, family_i, genus_i)
        hierarchy_name_map[klass] = (group, family, genus)

    classes = [class_index_to_name[i] for i in range(len(class_index_to_name))]
    groups = [group_index_to_name[i] for i in range(len(group_index_to_name))]
    families = [family_index_to_name[i] for i in range(len(family_index_to_name))]
    genuses = [genus_index_to_name[i] for i in range(len(genus_index_to_name))]

    return LabelHierarchy(
        classes=classes,
        groups=groups,
        families=families,
        genuses=genuses,
        class_map=class_name_to_index,
        group_map=group_name_to_index,
        family_map=family_name_to_index,
        genus_map=genus_name_to_index,
        index_map=hierarchy_index_map,
        name_map=hierarchy_name_map,
    )
