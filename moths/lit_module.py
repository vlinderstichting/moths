from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import MISSING
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss


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
        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val-loss", loss)
        return loss

    def validation_epoch_end(self, outputs: List[Dict[str, Tensor]]) -> None:
        pass

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> STEP_OUTPUT:
        pass

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
