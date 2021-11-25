from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from torch import tensor, Tensor
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import find_classes

from moths.label_hierarchy import LabelHierarchy


class LabelHierarchyImageFolder(ImageFolder):
    def __init__(
        self,
        root: Path,
        hierarchy: LabelHierarchy,
        transform: Optional[Callable] = None,
    ) -> None:
        """Override to use a predefined label hierarchy that is consistent over multiple
         sources.

        The `ImageFolder` uses `find_classes` to determine what classes are present.
        The line `classes, class_to_idx = self.find_classes(self.root)` is called in the
        __init__.

        Overload `find_classes` to return a mapping that is derived from the label
        hierarchy, such that it is consistent with that hierarchy.

        Additionally, create a target transform that expands the class (species) index
        to a tuple of class, group, family, and genus, indices.
        """
        self.class_to_idx_ = hierarchy.class_map

        def tfs(target_i: int) -> Tensor:
            out = tensor([target_i, *hierarchy.index_map[target_i]]).long()
            return out

        super(LabelHierarchyImageFolder, self).__init__(
            str(root),
            transform=transform,
            target_transform=tfs,
        )

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        classes, _ = find_classes(self.root)
        # use the pre determined class mapping, if cannot find assign it to other (0)
        class_to_idx = {c: (self.class_to_idx_[c] if c in self.class_to_idx_ else 0) for c in classes}
        return classes, class_to_idx
