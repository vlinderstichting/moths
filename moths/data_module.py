import logging
from dataclasses import dataclass
from os.path import abspath
from typing import Callable, List, Optional, Tuple

import pytorch_lightning as pl
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, transforms

from moths.datasets import (ConcatImageFolderDataset, FakeData,
                            LabelMapImageFolder)
from moths.path import resolve_original_path

log = logging.getLogger(__name__)


@dataclass
class DataConfig(DictConfig):
    data_paths: List[str]
    valid_path_file: str = "test_data/valid-paths.txt"
    label_map_file: str = "test_data/label-map.csv"
    test_fraction: float = 0.1

    output_size: Tuple[int, int] = (224, 224)

    batch_size: int = 1

    num_workers: int = 0
    pin_memory: bool = False

    fake_data: bool = False


class DataModule(pl.LightningDataModule):
    def __init__(self, config: DataConfig):
        super().__init__()
        self.config = config

    def prepare_data(self):
        valid_data_path = resolve_original_path(self.config.valid_path_file)
        log.info(f"Using valid data {str(valid_data_path)} ...")

        with valid_data_path.open("r") as f:
            self.valid_paths = set([p.strip() for p in f.readlines()])

        label_map_path = resolve_original_path(self.config.label_map_file)
        log.info(f"Using label map {str(label_map_path)} ...")

        with label_map_path.open("r") as f:
            lines = [line.strip() for line in f.readlines()]

        self.label_map = {l.split(",")[1]: int(l.split(",")[0]) for l in lines}
        self.num_classes = len(self.label_map)

    @property
    def _train_transform(self) -> Callable:
        # TODO: parameterize via config
        # Maybe possible to instaniate the transforms directly from config?
        tfs = []
        tfs.extend([transforms.RandomCrop(200), transforms.ToTensor()])

        return Compose(tfs)

    @property
    def _test_transform(self) -> Callable:
        # TODO: parameterize via config
        # Maybe possible to instaniate the transforms directly from config?
        tfs = []
        tfs.extend([transforms.CenterCrop(100), transforms.ToTensor()])

        return Compose(tfs)

    def _full_dataset(self, transform: Optional[Callable]) -> ConcatImageFolderDataset:
        datasets = []
        log.info(
            f"Using {[str(resolve_original_path(p)) for p in self.config.data_paths]}."
        )
        for data_path in self.config.data_paths:
            if self.config.fake_data:
                log.warning("Using fake data!")
                datasets.append(
                    FakeData(
                        size=self.config.fake_data, transform=transform, num_classes=2
                    )
                )
            else:
                data_path = resolve_original_path(data_path)
                log.debug(f"Scanning {str(data_path)} ...")
                datasets.append(
                    LabelMapImageFolder(
                        str(data_path),
                        label_map=self.label_map,
                        is_valid_file=lambda p: abspath(p) in self.valid_paths,
                        transform=transform,
                    )
                )

        return ConcatImageFolderDataset(datasets=datasets)

    def setup(self, stage: Optional[str] = None):

        full_train_dataset = self._full_dataset(self._train_transform)
        full_test_dataset = self._full_dataset(self._test_transform)

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
