from typing import Callable, Dict, List, Optional, Tuple

from torch import tensor
from torch.utils.data import ConcatDataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import find_classes

from moths.label_hierarchy import LabelHierarchy


class LabelHierarchyImageFolder(ImageFolder):
    def __init__(
        self,
        root: str,
        hierarchy: LabelHierarchy,
        transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        """Override to use a predefined label hierarchy that is consistent over multiple
         sources.

        The `ImageFolder` uses `find_classes` to determine what classes are present.
        The line `classes, class_to_idx = self.find_classes(self.root)` is called in the
        __init__.

        Overload `find_classes` to return a mapping that is derived from the label
        hierarchy, such that it is consistent across multiple data sources.

        Additionally, create a target transform that expands the class (species) index
        to a tuple of class, group, family, and genus, indices.
        """
        self.class_to_idx = hierarchy.class_map
        super(LabelHierarchyImageFolder, self).__init__(
            root,
            transform=transform,
            is_valid_file=is_valid_file,
            target_transform=lambda x: tensor([x, *hierarchy.index_map[x]]).long(),
        )

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        classes, _ = find_classes(self.root)
        class_to_idx = {k: v for k, v in self.class_to_idx.items() if k in classes}
        return classes, class_to_idx


class ConcatImageFolderDataset(ConcatDataset):
    """Add targets property that is needed for stratification.

    Not type safe since `ConcatDataset` also accepts datasets that do not have the
    targets property, so there is a risk of getting a `AttributeError` at run time.
    """

    @property
    def targets(self):
        return sum([d.targets for d in self.datasets], [])
