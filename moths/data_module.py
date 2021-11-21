import logging
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from torch import tensor
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose

from moths.config import resolve_config_path
from moths.label_hierarchy import hierarchy_from_path

log = logging.getLogger(__name__)


@dataclass
class DataConfig(DictConfig):
    data_path: str

    train_transforms: List[Any]
    test_transforms: List[Any]

    label_hierarchy_file: str
    test_fraction: float

    batch_size: int

    num_workers: int
    pin_memory: bool


class DataModule(pl.LightningDataModule):
    def __init__(self, config: DataConfig):
        """

        Args:
            config:
        """
        super().__init__()
        self.config = config

        train_tfs_instantiated = [instantiate(c) for c in config.train_transforms]
        self._train_transforms = Compose(train_tfs_instantiated)

        test_tfs_instantiated = [instantiate(c) for c in config.test_transforms]
        self._test_transforms = Compose(test_tfs_instantiated)

        label_hierarchy_path = resolve_config_path(self.config.label_hierarchy_file)
        self.label_hierarchy = hierarchy_from_path(label_hierarchy_path)

    def _full_dataset(self, transform: Optional[Callable]) -> ImageFolder:
        return ImageFolder(
            self.config.data_path,
            transform=transform,
            target_transform=lambda x: tensor(
                [x, *self.label_hierarchy.index_map[x]]
            ).long(),
        )

    def setup(self, stage: Optional[str] = None):
        # setup the full dataset twice, because we need different transforms
        # very little additional IO since the datasets are lazy, and during init they
        # only scan for the file names
        full_train_dataset = self._full_dataset(self._train_transforms)
        full_val_test_dataset = self._full_dataset(self._test_transforms)

        full_indices = list(range(len(full_train_dataset)))

        # always do equal size for test and val
        val_test_fraction = 2 * self.config.test_fraction
        # subtract the part for test and val from the total for train fraction
        train_fraction = 1 - val_test_fraction

        # targets are the same for both full datasets
        full_targets = full_train_dataset.targets

        train_indices, val_test_indices = train_test_split(
            full_indices,
            train_size=train_fraction,
            test_size=val_test_fraction,
            stratify=full_targets,
        )

        # get val test targets to stratify them as well
        val_test_targets = [full_targets[x] for x in val_test_indices]

        val_indices, test_indices = train_test_split(
            val_test_indices,
            train_size=0.5,
            test_size=0.5,
            stratify=val_test_targets,
        )

        # create subsets (from the correct full datasets) with the indices
        self.train_dataset = Subset(full_train_dataset, train_indices)
        self.val_dataset = Subset(full_val_test_dataset, val_indices)
        self.test_dataset = Subset(full_val_test_dataset, test_indices)

        log.debug(
            f"stratified a train, val, and test datasets with "
            f"{len(self.train_dataset)}, {len(self.val_dataset)}, "
            f"and {len(self.test_dataset)} samples, respectively"
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size * 2,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size * 2,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            shuffle=False,
            drop_last=False,
        )

    def predict_dataloader(self) -> DataLoader:
        pass
