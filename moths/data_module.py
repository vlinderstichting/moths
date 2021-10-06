import logging
from dataclasses import dataclass
from os.path import abspath
from typing import Any, Callable, List, Optional

import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose

from moths.config import resolve_config_path
from moths.datasets import (ConcatImageFolderDataset, FakeData,
                            LabelMapImageFolder)

log = logging.getLogger(__name__)


@dataclass
class DataConfig(DictConfig):
    data_paths: List[str]

    train_transforms: List[Any]
    test_transforms: List[Any]

    valid_path_file: str
    label_map_file: str
    test_fraction: float

    batch_size: int

    num_workers: int
    pin_memory: bool

    fake_data: bool


class DataModule(pl.LightningDataModule):
    def __init__(self, config: DataConfig):
        super().__init__()
        self.config = config

        train_tfs_instantiated = [instantiate(c) for c in config.train_transforms]
        self._train_transforms = Compose(train_tfs_instantiated)

        test_tfs_instantiated = [instantiate(c) for c in config.test_transforms]
        self._test_transforms = Compose(test_tfs_instantiated)

        if self.config.fake_data:
            log.info("Using fake data!")
        else:
            log.info(
                f"Using {[str(resolve_config_path(p)) for p in self.config.data_paths]}."
            )

    def prepare_data(self):
        valid_data_path = resolve_config_path(self.config.valid_path_file)
        log.info(f"Using valid data {str(valid_data_path)} ...")

        with valid_data_path.open("r") as f:
            self.valid_paths = set([p.strip() for p in f.readlines()])

        label_map_path = resolve_config_path(self.config.label_map_file)
        log.info(f"Using label map {str(label_map_path)} ...")

        with label_map_path.open("r") as f:
            lines = [line.strip() for line in f.readlines()]

        self.label_map = {l.split(",")[1]: int(l.split(",")[0]) for l in lines}
        self.num_classes = len(self.label_map)

    def _full_dataset(self, transform: Optional[Callable]) -> ConcatImageFolderDataset:
        datasets = []

        if self.config.fake_data:
            datasets = [
                FakeData(transform=transform, num_classes=self.num_classes)
                for _ in range(3)
            ]
        else:
            for data_path in self.config.data_paths:
                data_path = resolve_config_path(data_path)
                log.debug(f"Scanning {str(data_path)} ...")
                ds = LabelMapImageFolder(
                    str(data_path),
                    label_map=self.label_map,
                    is_valid_file=lambda p: abspath(p) in self.valid_paths,
                    transform=transform,
                )
                datasets.append(ds)

        return ConcatImageFolderDataset(datasets=datasets)

    def setup(self, stage: Optional[str] = None):
        full_train_dataset = self._full_dataset(self._train_transforms)
        full_test_dataset = self._full_dataset(self._test_transforms)

        all_indices = list(range(len(full_train_dataset)))

        val_test_fraction = 2 * self.config.test_fraction
        train_fraction = 1 - val_test_fraction

        targets = full_train_dataset.targets

        train_indices, val_test_indices = train_test_split(
            all_indices,
            train_size=train_fraction,
            test_size=val_test_fraction,
            stratify=targets,
        )

        val_test_targets = [targets[x] for x in val_test_indices]

        val_indices, test_indices = train_test_split(
            val_test_indices,
            train_size=0.5,
            test_size=0.5,
            stratify=val_test_targets,
        )

        self.train_dataset = Subset(full_train_dataset, train_indices)
        self.val_dataset = Subset(full_test_dataset, val_indices)
        self.test_dataset = Subset(full_test_dataset, test_indices)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size * 2,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size * 2,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            shuffle=False,
            drop_last=False,
        )
