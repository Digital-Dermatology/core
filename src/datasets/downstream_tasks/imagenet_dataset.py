import os
from glob import glob
from pathlib import Path
from typing import Union

import pandas as pd
import torch
from PIL import Image

from ....src.datasets.base_dataset import BaseDataset


class ImageNetDataset(BaseDataset):
    """ImageNet dataset."""

    IMG_COL = "img_path"
    LBL_COL = "class"

    def __init__(
        self,
        dataset_dir: Union[str, Path] = "data/imagenet-sample-images/",
        transform=None,
        val_transform=None,
        return_path: bool = False,
        **kwargs,
    ):
        """
        Initializes the dataset.

        Sets the correct path for the needed arguments.

        Parameters
        ----------
        dataset_dir : str
            Directory with all the images.
        transform : Union[callable, optional]
            Optional transform to be applied to the images.
        val_transform : Union[callable, optional]
            Optional transform to be applied to the images when in validation mode.
        return_path : bool
            If the path of the image should be returned or not.
        """
        super().__init__(transform=transform, val_transform=val_transform, **kwargs)
        # check if the dataset path exists
        self.dataset_dir = Path(dataset_dir)
        if not self.dataset_dir.exists():
            raise ValueError(f"Image path must exist, path: {self.dataset_dir}")

        # create dicts for retreiving imgs and masks
        imgs_path_dict = {
            os.path.splitext(os.path.basename(x))[0]: x
            for x in glob(os.path.join(dataset_dir, "*", "*.jpeg"))
        }

        # create the metadata dataframe
        self.meta_data = pd.DataFrame()
        for img_name, img_path in imgs_path_dict.items():
            label = "_".join(img_name.split("_")[1:])
            s_item = pd.Series([img_name, img_path, label])
            self.meta_data = pd.concat([self.meta_data, pd.DataFrame(s_item).T])
        self.meta_data.columns = ["img_name", self.IMG_COL, self.LBL_COL]
        self.meta_data.reset_index(drop=True, inplace=True)
        int_lbl, lbl_mapping = pd.factorize(self.meta_data[self.LBL_COL])
        self.meta_data["lbl_class"] = int_lbl
        # global configs
        self.return_path = return_path
        self.classes = list(lbl_mapping)
        self.n_classes = len(self.classes)

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.meta_data.loc[self.meta_data.index[idx], self.IMG_COL]
        image = Image.open(img_name)
        image = image.convert("RGB")
        if self.transform and self.training:
            image = self.transform(image)
        elif self.val_transform and not self.training:
            image = self.val_transform(image)

        diagnosis = self.meta_data.loc[self.meta_data.index[idx], "lbl_class"]
        if self.return_path:
            return image, img_name, int(diagnosis)
        else:
            return image, int(diagnosis)
