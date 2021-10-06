import bisect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import ConcatDataset, Dataset
from torchvision.datasets import ImageFolder, VisionDataset
from torchvision.datasets.folder import find_classes
from torchvision.transforms import transforms


class LabelMapImageFolder(ImageFolder):
    def __init__(
        self,
        root: str,
        label_map: Dict[str, int],
        transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        """Override to use a predefined label map that is consistent over multiple sources."""
        self.label_map = label_map
        super(LabelMapImageFolder, self).__init__(
            root, transform=transform, is_valid_file=is_valid_file
        )

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        classes, _ = find_classes(self.root)
        label_map = {k: v for k, v in self.label_map.items() if k in classes}
        return classes, label_map


class FakeData(VisionDataset):
    def __init__(
        self,
        size: int = 1000,
        image_size: Tuple[int, int, int] = (3, 512, 512),
        num_classes: int = 10,
        transform: Optional[Callable] = None,
    ) -> None:
        """Reimplements FakeData to have predetermined targets."""
        super(FakeData, self).__init__("", transform=transform)
        self.size = size
        self.num_classes = num_classes
        self.image_size = image_size

        self.targets = list(
            torch.randint(
                low=0,
                high=self.num_classes,
                size=(size,),
                dtype=torch.long,
            )
        )

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        rng_state = torch.get_rng_state()
        torch.set_rng_state(rng_state)
        torch.manual_seed(index)

        img = torch.randn(*self.image_size)

        img = transforms.ToPILImage()(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, self.targets[index]

    def __len__(self) -> int:
        return self.size


class ConcatImageFolderDataset(Dataset):
    def __init__(self, datasets: List[Union[ImageFolder, FakeData]]):
        """Stricter implementation of ConcatDataset for ImageFolders such that it can use targets."""
        super(ConcatImageFolderDataset, self).__init__()
        self.datasets = list(datasets)  # make super sure it is a list for order
        self.cumulative_sizes = ConcatDataset.cumsum(self.datasets)

    def __getitem__(self, idx: int) -> Tuple[Any, ...]:
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    def __len__(self):
        return sum(len(d) for d in self.datasets)

    @property
    def targets(self):
        return sum([d.targets for d in self.datasets], [])
