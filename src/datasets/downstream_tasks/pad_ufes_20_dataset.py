import os
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import torch
from PIL import Image

from ....src.datasets.base_dataset import BaseDataset


class PADLabel(Enum):
    DISEASE = "diagnostic"


class PADUFES20Dataset(BaseDataset):
    """PAD-UFES-20 dataset."""

    IMG_COL = "img_id"
    LBL_COL = "diagnostic"

    def __init__(
        self,
        csv_file: Union[str, Path] = "data/PAD-UFES-20/metadata.csv",
        root_dir: Union[str, Path] = "data/PAD-UFES-20/images/",
        transform=None,
        val_transform=None,
        label_col: PADLabel = PADLabel.DISEASE,
        return_path: bool = False,
        data_quality_issues_list: Optional[Union[str, Path]] = None,
        return_embedding: bool = False,
        **kwargs,
    ):
        """
        Initializes the dataset.

        Sets the correct path for the needed arguments.

        Parameters
        ----------
        csv_file : str
            Path to the csv file with metadata, including annotations.
        root_dir : str
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
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise ValueError(f"Image path must exist, path: {self.root_dir}")
        # load the metadata and encode the label
        self.meta_data = pd.DataFrame(pd.read_csv(self.csv_file))
        class_mapper = {
            "BCC": "Basal Cell Carcinoma",
            "SCC": "Squamous Cell Carcinoma",
            "ACK": "Actinic Keratosis",
            "SEK": "Seborrheic Keratosis",
            "BOD": "Bowen’s disease",
            "MEL": "Melanoma",
            "NEV": "Nevus",
        }
        self.meta_data[self.LBL_COL] = self.meta_data[self.LBL_COL].apply(
            class_mapper.get
        )
        int_lbl, lbl_mapping = pd.factorize(self.meta_data[self.LBL_COL])
        self.meta_data[self.LBL_COL + "_name"] = self.meta_data[self.LBL_COL]
        self.meta_data[self.LBL_COL] = int_lbl
        # remove data quality issues if file is given
        self.remove_data_quality_issues(data_quality_issues_list)
        self.meta_data.reset_index(drop=True, inplace=True)
        # global configs
        self.LBL_COL = label_col.value
        self.return_path = return_path
        self.return_embedding = return_embedding
        self.classes = (
            list(lbl_mapping)
            if label_col == PADLabel.DISEASE
            else ["benign", "malignant"]
        )
        self.n_classes = len(self.classes)

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.meta_data.loc[self.meta_data.index[idx], self.IMG_COL]
        img_name = os.path.join(self.root_dir, img_path)
        diagnosis = self.meta_data.loc[self.meta_data.index[idx], self.LBL_COL]

        if self.return_embedding:
            embedding = self.meta_data.loc[self.meta_data.index[idx], "embedding"]
            if self.return_path:
                return embedding, img_name, int(diagnosis)
            else:
                return embedding, int(diagnosis)

        image = Image.open(img_name)
        image = image.convert("RGB")
        if self.transform and self.training:
            image = self.transform(image)
        elif self.val_transform and not self.training:
            image = self.val_transform(image)

        if self.return_path:
            return image, img_name, int(diagnosis)
        else:
            return image, int(diagnosis)
