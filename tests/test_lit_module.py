from typing import Dict, List

import pytest
import torch
from hydra import compose, initialize_config_dir, initialize_config_module
from omegaconf import DictConfig
from torch import nn, tensor

from moths.label_hierarchy import LabelHierarchy
from moths.lit_module import LitConfig, LitModule
from moths.model import Model
from tests.test_integration import HYDRA_CONFIG_PATH


@pytest.fixture
def default_label_hierarchy():
    hierarchy: Dict[str, Dict[str, Dict[str, List[str]]]] = {
        "group_a": {
            "family_a": {
                "genus_a": ["species_a", "species_b"],
                "genus_b": ["species_c", "species_d"],
            },
            "family_b": {
                "genus_b": ["species_e", "species_f"],
                "genus_c": ["species_g", "species_h"],
            },
        },
        "group_b": {
            "family_c": {
                "genus_e": ["species_i", "species_j"],
                "genus_f": ["species_k", "species_l"],
            },
            "family_d": {
                "genus_g": ["species_m", "species_n"],
                "genus_h": ["species_o", "species_p"],
            },
        },
    }

    classes = []
    groups = []
    families = []
    genuses = []

    class_map = {}
    group_map = {}
    family_map = {}
    genus_map = {}

    index_map = {}
    name_map = {}

    cls_i = 0
    grp_i = 0
    fam_i = 0
    gen_i = 0

    for grp_n in hierarchy.keys():
        groups.append(grp_n)
        group_map[grp_n] = grp_i

        for fam_n in hierarchy[grp_n].keys():
            families.append(fam_n)
            family_map[fam_n] = fam_i

            for gen_n in hierarchy[grp_n][fam_n].keys():
                genuses.append(gen_n)
                genus_map[gen_n] = gen_i

                for cls_n in hierarchy[grp_n][fam_n][gen_n]:
                    classes.append(cls_n)
                    class_map[cls_n] = cls_i

                    index_map[cls_i] = (grp_i, fam_i, gen_i)
                    name_map[cls_n] = (grp_n, fam_n, gen_n)

                    cls_i += 1
                gen_i += 1
            fam_i += 1
        grp_i += 1

    return LabelHierarchy(
        classes=classes,
        groups=groups,
        families=families,
        genuses=genuses,
        class_map=class_map,
        group_map=group_map,
        family_map=family_map,
        genus_map=genus_map,
        index_map=index_map,
        name_map=name_map,
    )


def test_a(default_label_hierarchy):
    a = default_label_hierarchy


def test_model(default_label_hierarchy):
    initialize_config_dir(config_dir=HYDRA_CONFIG_PATH)
    config = compose(
        config_name="config",
        overrides=[
            f"model.pretrained=False",  # to prevent downloading in the test
        ],
    )


    model = Model(config.model, default_label_hierarchy)
    pass


def test_lit_module_loss_configuration():
    initialize_config_dir(HYDRA_CONFIG_PATH)
    config = compose(
        config_name="config",
    )
    config = config.lit

    lit_module = LitModule(config, nn.Module())
    loss_fn = lit_module.loss_fn

    # batch size 2
    # level 0: 5 classes
    # level 1: 4 classes
    # level 2: 3 classes
    # level 3: 2 classes

    y_hat = (
        torch.rand(2, 16),
        torch.rand(2, 2),
        torch.rand(2, 4),
        torch.rand(2, 8),
    )

    y = tensor(
        [
            [0, 15],
            [0, 1],
            [0, 3],
            [0, 7],
        ]
    )

    loss = loss_fn(y_hat, y)
    pass
