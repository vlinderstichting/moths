import pickle
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import typer
from torch import Tensor, tensor
from torch.nn import functional as F

from moths.label_hierarchy import LabelHierarchy
from moths.model import Model

tau_normalization_app = typer.Typer()


def shot_acc(
    preds, labels, train_data, many_shot_thr=100, low_shot_thr=20, acc_per_cls=False
):
    if isinstance(train_data, np.ndarray):
        training_labels = np.array(train_data).astype(int)
    else:
        training_labels = np.array(train_data.dataset.labels).astype(int)

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError("Type ({}) of preds not supported".format(type(preds)))
    train_class_count = []
    test_class_count = []
    class_correct = []
    for l in np.unique(labels):
        train_class_count.append(len(training_labels[training_labels == l]))
        test_class_count.append(len(labels[labels == l]))
        class_correct.append((preds[labels == l] == labels[labels == l]).sum())

    many_shot = []
    median_shot = []
    low_shot = []
    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            many_shot.append((class_correct[i] / test_class_count[i]))
        elif train_class_count[i] < low_shot_thr:
            low_shot.append((class_correct[i] / test_class_count[i]))
        else:
            median_shot.append((class_correct[i] / test_class_count[i]))

    if len(many_shot) == 0:
        many_shot.append(0)
    if len(median_shot) == 0:
        median_shot.append(0)
    if len(low_shot) == 0:
        low_shot.append(0)

    if acc_per_cls:
        class_accs = [c / cnt for c, cnt in zip(class_correct, test_class_count)]
        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot), class_accs
    else:
        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot)


def cos_similarity(A, B):
    normB = torch.norm(B, 2, 1, keepdim=True)
    B = B / normB
    AB = torch.mm(A, B.t())

    return AB


def linear_classifier(inputs, weights, bias):
    return torch.addmm(bias, inputs, weights.t())


def logits2preds(logits, labels):
    _, nns = logits.max(dim=1)
    preds = np.array([labels[i] for i in nns])
    return preds


def preds2accs(preds, testset, trainset):
    many, median, low, cls_accs = shot_acc(
        preds, testset["labels"], trainset["labels"], acc_per_cls=True
    )
    top1_all = np.mean(cls_accs)
    print(
        "{:.2f} \t {:.2f} \t {:.2f} \t {:.2f}".format(
            many * 100, median * 100, low * 100, top1_all * 100
        )
    )


def pnorm(weights, p):
    normB = torch.norm(weights, 2, 1)
    ws = weights.clone()
    for i in range(weights.size(0)):
        ws[i] = ws[i] / torch.pow(normB[i], p)
    return ws


def dotproduct_similarity(A, B):
    AB = torch.mm(A, B.t())

    return AB


@dataclass
class ModelConfig:
    zoo_name: str
    pretrained: bool


def strip_key(key):
    return key[6:]  # remove "model." from the start of the key


def print_acc(preds: Tensor, species_y: Tensor):
    classes = torch.unique(species_y)
    class_counts = [(klass, (species_y == klass).sum()) for klass in classes]

    low = 50
    high = 200

    classes_few = tensor([k for k, n in class_counts if n < low])
    classes_many = tensor([k for k, n in class_counts if n > high])
    classes_some = tensor([k for k, n in class_counts if n >= low and n <= high])

    out = f"\t overall: {(preds == species_y).sum() / len(species_y):.2f}"

    for classes_segment, name in [
        (classes_few, "few"),
        (classes_some, "some"),
        (classes_many, "many"),
    ]:
        mask = torch.isin(species_y, classes_segment)
        preds_masked = torch.masked_select(preds, mask)
        species_y_masked = torch.masked_select(species_y, mask)
        acc_masked = (preds_masked == species_y_masked).sum() / len(species_y_masked)
        out += f"\t {name}: {acc_masked:.2f}"

    print(out)


@tau_normalization_app.command()
def tau_normalization(path: Path) -> None:

    with (path / "label_hierarchy.pkl").open("rb") as f:
        label_hierarchy: LabelHierarchy = pickle.load(f)

    config = ModelConfig(zoo_name="efficientnet_b7", pretrained=True)

    ckpt = torch.load(str(path / "best.ckpt"), map_location=torch.device("cpu"))
    ckpt_new = OrderedDict(
        [(strip_key(key), value) for key, value in ckpt["state_dict"].items()]
    )

    model = Model(config, label_hierarchy)
    model.load_state_dict(ckpt_new)

    weights = model.fc_class.weight
    bias = model.fc_class.bias

    species_y = torch.Tensor(np.load(str(path / "arrays" / "0_species_y.npy")))
    species_logits = torch.Tensor(
        np.load(str(path / "arrays" / "0_species_logits.npy"))
    )
    features = torch.Tensor(np.load(str(path / "arrays" / "features.npy")))

    # todo: train without bias
    # todo: retrain the last layer

    logits = F.linear(features, weights, bias)
    logits = dotproduct_similarity(features, weights) + bias
    preds = torch.argmax(logits, dim=1)
    acc = (preds == species_y).sum() / len(species_y)
    print(f"baseline: {acc:.2f}")
    print_acc(preds, species_y)

    # breakpoint()

    for p in np.linspace(0, 2, 21):
        # print(p)
        ws = pnorm(weights, p)
        logits = dotproduct_similarity(features, ws) + bias
        preds = torch.argmax(logits, dim=1)
        print_acc(preds, species_y)
        # acc = (preds == species_y).sum() / len(species_y)
        # print(f"{p:.2f}: {acc:.2f}")
        # preds2accs(preds, testset, trainset)


if __name__ == "__main__":
    tau_normalization_app()
