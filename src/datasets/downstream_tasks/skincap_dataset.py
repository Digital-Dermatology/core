import os
from enum import Enum
from glob import glob
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import torch
from PIL import Image

from ....src.datasets.base_dataset import BaseDataset


class SkinCapDataset(BaseDataset):
    """SkinCap dataset."""

    IMG_COL = "skincap_file_path"

    def __init__(
        self,
        csv_file: Union[str, Path] = "data/SkinCap/ddi_metadata.csv",
        dataset_dir: Union[str, Path] = "data/SkinCap/images",
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
        csv_file : str
            Path to the csv file with metadata, including annotations.
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
        self.csv_file = Path(csv_file)
        if not self.csv_file.exists():
            raise ValueError(f"CSV metadata path must exist, path: {self.csv_file}")
        self.dataset_dir = Path(dataset_dir)
        if not self.dataset_dir.exists():
            raise ValueError(f"Image path must exist, path: {self.dataset_dir}")
        # transform the dataframe for better loading
        imageid_path_dict = {
            os.path.basename(x): x
            for x in glob(os.path.join(self.dataset_dir, "*", "*.png"))
        }
        # load the metadata
        self.meta_data = pd.DataFrame(pd.read_csv(csv_file, index_col=0))
        self.meta_data["path"] = self.meta_data[self.IMG_COL].map(imageid_path_dict.get)

        self.meta_data["description"] = self.meta_data['caption_zh_polish_en']

        # global configs
        self.return_path = return_path
        self.IMG_COL = 'path'
        self.LBL_COL = 'description'
        self.classes = None
        self.n_classes = None

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

        description = self.meta_data.loc[self.meta_data.index[idx], self.LBL_COL]
        if self.return_path:
            return image, img_name, description
        else:
            return image, description
