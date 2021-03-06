import logging
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

import numpy as np
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose

from moths.config import resolve_config_path
from moths.datasets import LabelHierarchyImageFolder
from moths.label_hierarchy import label_hierarchy_from_file

log = logging.getLogger("MOTHS")


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

    min_samples: int
    weighted_sampling: bool
    weighted_sampling_fraction: float


class DataModule(pl.LightningDataModule):
    def __init__(self, config: DataConfig):
        super().__init__()
        self.config = config

        train_tfs_instantiated = [instantiate(c) for c in config.train_transforms]
        self._train_transforms = Compose(train_tfs_instantiated)

        test_tfs_instantiated = [instantiate(c) for c in config.test_transforms]
        self._test_transforms = Compose(test_tfs_instantiated)

        label_hierarchy_path = resolve_config_path(config.label_hierarchy_file)
        data_source_path = resolve_config_path(config.data_path)
        self.label_hierarchy = label_hierarchy_from_file(
            label_hierarchy_path, data_source_path, config.min_samples
        )

        log.info(
            f"Final class count: "
            f"{len(self.label_hierarchy.classes) - 1} "
            f"{len(self.label_hierarchy.groups) - 1} "
            f"{len(self.label_hierarchy.families) - 1} "
            f"{len(self.label_hierarchy.genuses) - 1}."
        )

        self.batch_size = config.batch_size

    def _full_dataset(self, transform: Optional[Callable]) -> ImageFolder:
        return LabelHierarchyImageFolder(
            resolve_config_path(self.config.data_path),
            hierarchy=self.label_hierarchy,
            transform=transform,
        )

    def setup(self, stage: Optional[str] = None):
        # setup the full dataset twice, because we need different transforms
        # very little additional IO since the datasets are lazy, and during init they
        # only scan for the file names
        full_train_dataset = self._full_dataset(self._train_transforms)
        full_val_test_dataset = self._full_dataset(self._test_transforms)

        log.debug("created train and val_test datasets")

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

        if self.config.weighted_sampling and self.config.weighted_sampling_fraction > 0:
            train_targets = [full_targets[x] for x in train_indices]
            targets_unique, targets_counts = np.unique(
                train_targets, return_counts=True
            )
            targets_weight_per_target = targets_counts.sum() / targets_counts
            targets_weight_per_target = (
                targets_weight_per_target ** self.config.weighted_sampling_fraction
            )
            target_weight_map = {
                targets_unique[i]: targets_weight_per_target[i]
                for i in range(len(targets_unique))
            }
            target_weights = [target_weight_map[t] for t in train_targets]
            self.train_sampler = WeightedRandomSampler(
                target_weights, len(targets_unique), replacement=True
            )
        else:
            self.train_sampler = None

        # create subsets (from the correct full datasets) with the indices
        self.train_dataset = Subset(full_train_dataset, train_indices)
        self.val_dataset = Subset(full_val_test_dataset, val_indices)
        self.test_dataset = Subset(full_val_test_dataset, test_indices)

        log.info(
            f"stratified a train, val, and test datasets with "
            f"{len(self.train_dataset)}, {len(self.val_dataset)}, "
            f"and {len(self.test_dataset)} samples, respectively"
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            shuffle=self.train_sampler is None,
            drop_last=True,
            sampler=self.train_sampler,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            shuffle=False,
            drop_last=False,
        )

    def predict_dataloader(self) -> DataLoader:
        pass
