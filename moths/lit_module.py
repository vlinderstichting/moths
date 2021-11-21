from dataclasses import dataclass
from functools import cached_property
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import MISSING
from torch import Tensor, nn
from torchmetrics import Metric

from moths.label_hierarchy import LABELS
from moths.model import Model

PRED_TRUTH_PAIR = Tuple[Tensor, Tensor]  # NC, N
LABEL_PAIRS = Tuple[PRED_TRUTH_PAIR, PRED_TRUTH_PAIR, PRED_TRUTH_PAIR, PRED_TRUTH_PAIR]


@dataclass
class LitConfig:
    optimizer: Any = MISSING
    scheduler: Optional[Any] = None
    scheduler_interval: str = "step"

    loss: Any = MISSING
    loss_weights: Tuple[float, float, float, float] = (1, 1, 1, 1)

    train_metrics: List[Any] = MISSING
    test_metrics: List[Any] = MISSING


class LitModule(pl.LightningModule):
    """
    Everywhere in this module it is assumed that 4 labels are being predicted and
    scored, namely; species, group, family, and genus, in that order.
    """

    def __init__(self, config: LitConfig, model: Model) -> None:
        super(LitModule, self).__init__()
        self.config = config
        self.model = model

        # note same metrics for val and test
        # set, level, metrics
        self.metrics: Dict[str, Dict[str, List[Metric]]] = {
            "train": {
                l: [instantiate(c) for c in config.train_metrics] for l in LABELS
            },
            "val": {l: [instantiate(c) for c in config.test_metrics] for l in LABELS},
            "test": {l: [instantiate(c) for c in config.test_metrics] for l in LABELS},
        }

        for label in LABELS:
            for phase_name in ["train", "val", "test"]:
                for metric in self.metrics[phase_name][label]:
                    metric.to("cuda")

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.model.forward(x)

    @cached_property
    def loss_fn(self) -> Callable[[Tensor, Tensor], Tensor]:
        _loss_fn: nn.Module = instantiate(self.config.loss)
        weights = torch.tensor([self.config.loss_weights]).long().to(self.device)

        def _out_fn(y_hat: Tensor, y: Tensor) -> Tensor:
            losses = torch.stack([_loss_fn(y_hat[i], y[i]) for i in range(len(LABELS))])
            breakpoint()
            return torch.mean(losses * weights)

        return _out_fn

    def _transform_batch(self, batch: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        x, y = batch
        return x, y.T

    def _step(self, batch: Tuple[Tensor, Tensor], phase_name: str) -> Tensor:
        x, y = self._transform_batch(batch)
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)

        self.log(f"{phase_name}-loss", loss)

        with torch.no_grad():
            # assume same order
            for i, l in enumerate(LABELS):
                for metric in self.metrics[phase_name][l]:
                    y_hat_discrete = torch.argmax(y_hat[i], dim=1)
                    metric_out = metric(y_hat_discrete, y[i])
                    log_name = f"{phase_name}-{l}-{metric.__class__.__name__.lower()}"
                    self.log(log_name, metric_out)

        return loss

    def _clear_metrics(self, phase_name: str):
        for i, l in enumerate(LABELS):
            for metric in self.metrics[phase_name][l]:
                metric.reset()

    def _log_metric_compute(self, phase_name: str):
        # assume same order
        for i, l in enumerate(LABELS):
            for metric in self.metrics[phase_name][l]:
                log_name = f"epoch-{phase_name}-{l}-{metric.__class__.__name__.lower()}"
                self.log(log_name, metric.compute())

    def on_train_epoch_start(self) -> None:
        self._clear_metrics("train")

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        return self._step(batch, "train")

    def training_epoch_end(self, outputs: List[Tensor]) -> None:
        self._log_metric_compute("train")

    def on_validation_epoch_start(self) -> None:
        self._clear_metrics("val")

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        return self._step(batch, "val")

    def validation_epoch_end(self, outputs: List[Tensor]) -> None:
        self._log_metric_compute("val")

    def on_test_epoch_start(self) -> None:
        self._clear_metrics("test")

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        return self._step(batch, "test")

    def test_epoch_end(self, outputs: List[Tensor]) -> None:
        self._log_metric_compute("test")
        # # prediction come in as [num batches, batch size, num classes]
        # # labels come in as [num batches, batch size]
        #
        # preds = torch.stack([d["pred"] for d in outputs])
        # preds = torch.argmax(preds, dim=2)
        # preds = torch.flatten(preds).tolist()
        #
        # labels = torch.stack([d["label"] for d in outputs])
        # labels = torch.flatten(labels).tolist()
        #
        # wandb.log(
        #     {
        #         "class CM": wandb.plot.confusion_matrix(
        #             probs=None,
        #             y_true=labels,
        #             preds=preds,
        #             class_names=self.trainer.datamodule.class_names,
        #         )
        #     }
        # )

    def configure_optimizers(self):
        optimizer = instantiate(
            self.config.optimizer, params=self.model.parameters(), _convert_="partial"
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
