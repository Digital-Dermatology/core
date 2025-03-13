import os
from glob import glob
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import torch
from PIL import Image

from ....src.datasets.base_dataset import BaseDataset


class HAM10000Dataset(BaseDataset):
    """HAM10000 dataset."""

    IMG_COL = "image_id"
    LBL_COL = "dx"

    def __init__(
        self,
        csv_file: Union[str, Path] = "data/HAM10000/HAM10000_metadata.csv",
        dataset_dir: Union[str, Path] = "data/HAM10000/",
        transform=None,
        val_transform=None,
        return_path: bool = False,
        test_csv_file: Optional[Union[str, Path]] = None,
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
            os.path.splitext(os.path.basename(x))[0]: x
            for x in glob(os.path.join(self.dataset_dir, "*", "*.jpg"))
        }
        lesion_type_dict = {
            "nv": "Melanocytic nevi",
            "mel": "dermatofibroma",
            "bkl": "Benign keratosis-like lesions ",
            "bcc": "Basal cell carcinoma",
            "akiec": "Actinic keratoses",
            "vasc": "Vascular lesions",
            "df": "Dermatofibroma",
        }
        # load the metadata
        self.meta_data = pd.DataFrame(pd.read_csv(csv_file))
        self.meta_data["dataset_origin"] = "Train"
        if test_csv_file is not None:
            _test_meta_data = pd.DataFrame(pd.read_csv(test_csv_file))
            _test_meta_data["dataset_origin"] = "Test"
            self.meta_data = pd.concat([self.meta_data, _test_meta_data])
        self.meta_data["path"] = self.meta_data[self.IMG_COL].map(imageid_path_dict.get)
        self.meta_data["cell_type"] = self.meta_data[self.LBL_COL].map(
            lesion_type_dict.get
        )
        int_lbl, lbl_mapping = pd.factorize(self.meta_data["cell_type"])
        self.meta_data["cell_type_idx"] = int_lbl
        self.meta_data.dropna(subset=["path"], inplace=True)
        self.meta_data.reset_index(drop=True, inplace=True)
        self.LBL_COL = "cell_type_idx"
        # global configs
        self.return_path = return_path
        self.classes = list(lbl_mapping)
        self.n_classes = len(self.classes)

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.meta_data.loc[self.meta_data.index[idx], "path"]
        image = Image.open(img_name)
        image = image.convert("RGB")
        if self.transform and self.training:
            image = self.transform(image)
        elif self.val_transform and not self.training:
            image = self.val_transform(image)

        diagnosis = self.meta_data.loc[self.meta_data.index[idx], self.LBL_COL]
        if self.return_path:
            return image, img_name, int(diagnosis)
        else:
            return image, int(diagnosis)
