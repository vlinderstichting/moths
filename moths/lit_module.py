import itertools
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor, nn, tensor
from torchmetrics import Metric

from moths.label_hierarchy import LABELS, LabelHierarchy, get_classes_by_label
from moths.mix import mix_loss, mixup_batch
from moths.model import Model

log = logging.getLogger(__name__)

LABEL_OUTPUT = Tuple[Tensor, Tensor]  # logits (N,C,(2??)) and targets (N,)
BATCH_OUTPUT = Dict[
    str, Union[LABEL_OUTPUT, Tensor]
]  # one for every label and "loss" for loss

PHASES = ["train", "val", "test"]


class MixMode(Enum):
    MIXUP = "mixup"
    CUTMIX = "cutmix"


@dataclass
class MixConfig:
    mode: MixMode
    probability: float
    alpha: float


@dataclass
class LitConfig:
    loss: dict
    loss_weights: Tuple[float, float, float, float]

    metrics: List[Any]

    unfreeze_backbone_epoch_start: int
    unfreeze_backbone_epoch_duration: int
    unfreeze_backbone_percentage: float

    optimizer: Any
    scheduler: Optional[dict] = None
    scheduler_interval: str = "None"

    data_mix: Optional[MixConfig] = None


class LitModule(pl.LightningModule):
    """
    Everywhere in this module it is assumed that 4 labels are being predicted and
    scored, namely; species, group, family, and genus, in that order.
    """

    def __init__(
        self, config: LitConfig, model: Model, label_hierarchy: LabelHierarchy
    ) -> None:
        super(LitModule, self).__init__()
        self.config = config
        self.model = model
        self.label_hierarchy = label_hierarchy
        self.use_mix = config.data_mix is not None
        self.lr = config.optimizer.lr

        def instantiate_metric(config: DictConfig, label: str):
            num_classes = len(get_classes_by_label(label_hierarchy, label))
            return instantiate(config, num_classes=num_classes, _convert_="partial")

        # note same metrics for val and test
        # set, level, metrics
        self.metrics: Dict[str, Dict[str, List[Metric]]] = {
            phase: {
                l: [instantiate_metric(c, l) for c in config.metrics] for l in LABELS
            }
            for phase in PHASES
        }

        self._loss_fn: nn.Module = instantiate(self.config.loss)
        self._loss_weights = torch.tensor([self.config.loss_weights]).long()

        self._backbone_parameters = [
            p for p in model.backbone.parameters() if p.requires_grad
        ]
        self._freeze_backbone()

    def loss_fn(self, y_hat: Tensor, y: Tensor) -> Tensor:
        # torch.clone because otherwise it crashes, bug?!
        losses = [self._loss_fn(y_hat[i], torch.clone(y[i])) for i in [0, 1, 2, 3]]
        return torch.mean(torch.stack(losses) * self._loss_weights)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.model.forward(x)

    def _transform_batch(self, batch: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        x, y = batch
        return x, y.T

    def _step(self, batch: Tuple[Tensor, Tensor], phase: str) -> BATCH_OUTPUT:
        x, y = self._transform_batch(batch)

        if self.use_mix and phase == "train":
            x, ya, yb, l = mixup_batch(
                x, y, self.config.data_mix.probability, self.config.data_mix.alpha
            )
            y_hat = self.model(x)
            loss = mix_loss(self.loss_fn, y_hat, ya, yb, l)
        else:
            y_hat = self.model(x)
            loss = self.loss_fn(y_hat, y)

        self.log(f"{phase}-loss", loss.detach())

        # assume same order
        for i, l in enumerate(LABELS):
            y_hat_label = torch.argmax(y_hat[i], dim=1).detach()
            y_label = y[i].detach()
            for metric in self.metrics[phase][l]:
                log_value = metric(y_hat_label, y_label)
                if log_value is None:
                    continue
                log_name = f"{phase}-{l}-{metric.__class__.__name__.lower()}"

                if phase == "train":
                    self.log(log_name, log_value)

        out = {l: (y_hat[i].detach(), y[i].detach()) for i, l in enumerate(LABELS)}
        out["loss"] = loss
        out["size"] = tensor(x.size()[0], device=loss.device)

        return out

    def _clear_metrics(self, phase: str):
        for label in LABELS:
            for metric in self.metrics[phase][label]:
                metric.reset()

    def _log_metric_compute(self, phase: str):
        # assume same order
        for label in LABELS:
            for metric in self.metrics[phase][label]:
                log_value = metric.compute()
                if log_value is None:
                    continue
                log_name = f"epoch-{phase}-{label}-{metric.__class__.__name__.lower()}"
                self.log(log_name, log_value)

    def _log_epoch_loss(self, phase: str, outputs: List[BATCH_OUTPUT]):
        losses = torch.stack([o["loss"] for o in outputs])
        sizes = torch.stack([o["size"] for o in outputs])
        loss = (losses * sizes).sum() / sizes.sum()
        self.log(f"epoch-{phase}-loss", loss.detach())

    def _log_north_star(self, phase: str, outputs: List[BATCH_OUTPUT]):
        preds = (
            torch.concat([batch["species"][0] for batch in outputs])
            .detach()
            .cpu()
            .numpy()
        )
        targets = (
            torch.concat([batch["species"][1] for batch in outputs])
            .detach()
            .cpu()
            .numpy()
        )
        return None

    def _freeze_backbone(self):
        for p in self._backbone_parameters:
            p.requires_grad = False

        log.info("Frozen the complete backbone.")

    def _unfreeze_backbone(self, fraction: float):
        layer_ix = int(round((1 - fraction) * len(self._backbone_parameters)))

        # assumes that the parameters are in order of the network
        for p in self._backbone_parameters[layer_ix:]:
            p.requires_grad = True

        nb_layers = len(self._backbone_parameters) - layer_ix
        pt_layers = round(fraction * 100)
        log.info(
            f"Unfrozen {nb_layers}/{len(self._backbone_parameters)} ({pt_layers}%) of the last layers."
        )

        nb_frozen = sum([not p.requires_grad for p in self._backbone_parameters])
        pt_frozen = round((nb_frozen / len(self._backbone_parameters)) * 100)
        log.info(f"Currently {nb_frozen} ({pt_frozen}%) frozen")

    def _unfreeze_backbone_from_config(self):
        epoch_start = self.config.unfreeze_backbone_epoch_start
        epoch_end = epoch_start + self.config.unfreeze_backbone_epoch_duration

        final_fraction = self.config.unfreeze_backbone_percentage

        fraction = np.interp(
            self.current_epoch, [epoch_start - 1, epoch_end], [0, final_fraction]
        )
        self._unfreeze_backbone(float(fraction))

    def on_fit_start(self) -> None:
        for phase, label in itertools.product(PHASES, LABELS):
            for metric in self.metrics[phase][label]:
                metric.to(self.device)

        self._loss_weights.to(self.device)

    def on_train_epoch_start(self):
        self._clear_metrics("train")
        self._unfreeze_backbone_from_config()

    def training_step(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int
    ) -> BATCH_OUTPUT:
        return self._step(batch, "train")

    def training_epoch_end(self, outputs: List[BATCH_OUTPUT]):
        self._log_metric_compute("train")
        self._log_epoch_loss("train", outputs)
        self._log_north_star("train", outputs)

    def on_validation_epoch_start(self):
        self._clear_metrics("val")

    def validation_step(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int
    ) -> BATCH_OUTPUT:
        return self._step(batch, "val")

    def validation_epoch_end(self, outputs: List[BATCH_OUTPUT]):
        self._log_metric_compute("val")
        self._log_epoch_loss("val", outputs)
        self._log_north_star("val", outputs)

    def on_test_epoch_start(self):
        self._clear_metrics("test")

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> BATCH_OUTPUT:
        return self._step(batch, "test")

    def test_epoch_end(self, outputs: List[BATCH_OUTPUT]):
        self._log_metric_compute("test")
        self._log_epoch_loss("test", outputs)
        self._log_north_star("test", outputs)

    def configure_optimizers(self):
        optimizer = instantiate(
            self.config.optimizer,
            params=self.model.parameters(),
            _convert_="partial",
        )

        if self.config.scheduler is None:
            return optimizer

        scheduler = instantiate(
            self.config.scheduler, optimizer=optimizer, _convert_="partial"
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": self.config.scheduler_interval,
            },
        }
