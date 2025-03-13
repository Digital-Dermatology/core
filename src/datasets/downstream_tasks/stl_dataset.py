from typing import Any, Callable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision


class STL10Dataset(torchvision.datasets.STL10):
    """STL-10 dataset."""

    LBL_COL = "label"

    def __init__(
        self,
        root: str,
        split: str = "train",
        folds: Optional[int] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        **kwargs,
    ):
        super().__init__(
            root=root,
            split=split,
            folds=folds,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        self.meta_data = pd.DataFrame(
            np.arange(self.data.shape[0]),
            columns=["data index"],
        )
        self.meta_data["label"] = self.labels
        # global configs
        self.n_classes = len(self.classes)

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        rets = super().__getitem__(index=index)
        rets = (rets[0], "", rets[1])
        return rets

    @staticmethod
    def collate_fn(batch):
        return torch.utils.data.dataloader.default_collate(batch)
