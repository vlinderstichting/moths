import pickle
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import typer
from pycm import ConfusionMatrix

from moths.label_hierarchy import LabelHierarchy

matplotlib.use("AGG")


evaluate_predictions_app = typer.Typer()


def plot_cm(
    cm: ConfusionMatrix,
    path: Path,
    num_items: int,
    normalized: bool = False,
    class_name: Optional[str] = None,
) -> None:
    max_items_num_label = 20
    ax = cm.plot(
        normalized=normalized,
        one_vs_all=class_name is not None,
        class_name=class_name,
        number_label=num_items < max_items_num_label,
    )

    plt.xticks(rotation=90)

    size = int(np.interp(num_items, [2, 30], [10, 26]).round())

    padding_size = 3
    padding_fraction = padding_size / size

    plt.subplots_adjust(bottom=padding_fraction, left=padding_fraction)
    ax.figure.set_size_inches((size, size))

    if normalized:
        path = Path(f"{str(path)}-normalized")

    ax.figure.savefig(str(path.with_suffix(".png")))
    plt.close(ax.figure)


def plot_cms(
    label_hierarchy: LabelHierarchy,
    species_y: np.ndarray,
    species_y_hat: np.ndarray,
    family_y: np.ndarray,
    family_y_hat: np.ndarray,
    out_path: Path,
) -> None:
    cm_family = ConfusionMatrix(family_y, family_y_hat)
    cm_family.relabel(mapping={i: l for i, l in enumerate(label_hierarchy.families)})

    (out_path / "family").mkdir(exist_ok=True, parents=True)

    num_families = len(np.unique(family_y))

    plot_cm(cm_family, out_path / "family", num_items=num_families, normalized=False)
    plot_cm(cm_family, out_path / "family", num_items=num_families, normalized=True)

    for family_ix, family_name in enumerate(label_hierarchy.families):
        plot_cm(
            cm_family,
            out_path / "family" / family_name,
            num_items=num_families,
            normalized=True,
            class_name=family_name,
        )
        plot_cm(
            cm_family,
            out_path / "family" / family_name,
            num_items=num_families,
            normalized=False,
            class_name=family_name,
        )

        selection = family_y == family_ix
        spiecies_y_i = species_y[selection]
        spiecies_y_hat_i = species_y_hat[selection]

        unique_classes = np.unique(spiecies_y_i)
        unique_classes.sort()

        if len(unique_classes) == 1:
            print(f"{family_name} has only one species. Skipping ...")
            continue

        unique_classes_hat = np.unique(spiecies_y_hat_i)
        unique_classes_hat.sort()

        big_int = np.iinfo(np.int32).max
        mapping = {
            i: l for i, l in enumerate(label_hierarchy.classes) if i in unique_classes
        }

        if not np.array_equal(unique_classes, unique_classes_hat):
            mapping[big_int] = "Other families"

        spiecies_y_hat_i = np.array(
            [(y if y in unique_classes else big_int) for y in spiecies_y_hat_i]
        )

        cm_species_i = ConfusionMatrix(spiecies_y_i, spiecies_y_hat_i)
        cm_species_i.relabel(mapping=mapping)

        out_path_i = out_path / "families" / family_name

        out_path_i.parent.mkdir(exist_ok=True)

        num_species = len(mapping)

        plot_cm(cm_species_i, out_path_i, num_items=num_species, normalized=False)
        plot_cm(cm_species_i, out_path_i, num_items=num_species, normalized=True)


def print_worst_performing_classes(
    label_hierarchy: LabelHierarchy,
    species_y: np.ndarray,
    species_y_hat: np.ndarray,
) -> None:
    cm_species = ConfusionMatrix(species_y, species_y_hat)
    cm_species.relabel(mapping={i: l for i, l in enumerate(label_hierarchy.classes)})

    print(cm_species.stat(summary=True, class_name=[]))


@evaluate_predictions_app.command()
def evaluate_predictions(path: Path) -> None:

    with (path / "label_hierarchy.pkl").open("rb") as f:
        label_hierarchy = pickle.load(f)

    species_y = np.load(str(path / "arrays" / "0_species_y.npy"))
    species_y_hat = np.load(str(path / "arrays" / "0_species_y_hat.npy"))

    family_y = np.load(str(path / "arrays" / "2_family_y.npy"))
    family_y_hat = np.load(str(path / "arrays" / "2_family_y_hat.npy"))

    print_worst_performing_classes(label_hierarchy, species_y, species_y_hat)
    plot_cms(
        label_hierarchy, species_y, species_y_hat, family_y, family_y_hat, path / "cms"
    )


if __name__ == "__main__":
    evaluate_predictions_app()
