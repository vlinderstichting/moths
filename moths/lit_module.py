from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import MISSING
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from torchmetrics import Accuracy


@dataclass
class LitConfig:
    optimizer: Any = MISSING
    scheduler: Optional[Any] = None
    scheduler_interval: str = "step"


class LitModule(pl.LightningModule):
    def __init__(self, config: LitConfig, model: nn.Module) -> None:
        super(LitModule, self).__init__()
        self.config = config
        self.model = model

        self.train_acc = Accuracy()
        self.valid_acc = Accuracy()

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    @cached_property
    def loss_fn(self):
        return CrossEntropyLoss()

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train-loss", loss)
        self.train_acc(y_hat, y)
        self.log("train-acc", self.train_acc)
        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val-loss", loss)
        self.valid_acc(y_hat, y)
        self.log("val-acc", self.valid_acc)
        return loss

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Any]:
        x, y = batch
        y_hat = self.model(x)
        return {"pred": y_hat, "label": y}

    def test_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:

        # prediction come in as [num batches, batch size, num classes]
        # labels come in as [num batches, batch size]

        preds = torch.stack([d["pred"] for d in outputs])
        preds = torch.argmax(preds, dim=2)
        preds = torch.flatten(preds).tolist()

        labels = torch.stack([d["label"] for d in outputs])
        labels = torch.flatten(labels).tolist()

        wandb.log(
            {
                "class CM": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=labels,
                    preds=preds,
                    class_names=self.trainer.datamodule.class_names,
                )
            }
        )

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
