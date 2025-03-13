from pathlib import Path
from typing import Union

import pandas as pd
import torch
from PIL import Image

from ....src.datasets.base_dataset import BaseDataset


class PatchCamelyonDataset(BaseDataset):
    """Patch Camelyon dataset."""

    IMG_COL = "image_path"
    LBL_COL = "class_name"

    def __init__(
        self,
        dataset_dir: Union[str, Path] = "data/PatchCamelyon/",
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
        return_path : bool
            If the path of the image should be returned or not.
        """
        super().__init__(transform=transform, val_transform=val_transform, **kwargs)
        # check if the dataset path exists
        self.dataset_dir = Path(dataset_dir)
        if not self.dataset_dir.exists():
            raise ValueError(f"Image path must exist, path: {self.dataset_dir}")
        # load the metadata
        self.meta_data = pd.DataFrame()
        for meta_file in self.dataset_dir.glob("*_meta.csv"):
            origin = str(meta_file.stem).replace("camelyonpatch_level_2_split_", "")
            origin = origin.replace("_meta", "").capitalize()
            _df = pd.read_csv(meta_file, index_col=0)
            _df["origin"] = origin
            self.meta_data = pd.concat([self.meta_data, _df])
        self.meta_data[self.IMG_COL] = self.meta_data.apply(
            lambda row: str(
                self.dataset_dir
                / row["origin"]
                / f"{row['wsi']}__{str(row['coord_x'])}__{str(row['coord_y'])}.jpg"
            ),
            axis=1,
        )
        # one-hot encode label
        int_lbl, lbl_mapping = pd.factorize(self.meta_data["tumor_patch"])
        self.meta_data["class_name_idx"] = int_lbl
        self.meta_data.dropna(subset=[self.IMG_COL], inplace=True)
        self.meta_data.reset_index(drop=True, inplace=True)
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

        diagnosis = self.meta_data.loc[self.meta_data.index[idx], "class_name_idx"]
        if self.return_path:
            return image, img_name, int(diagnosis)
        else:
            return image, int(diagnosis)
