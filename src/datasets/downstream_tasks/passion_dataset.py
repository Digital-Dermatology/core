import re
from enum import Enum
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile

from ....src.datasets.generic_image_dataset import GenericImageDataset

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


class PASSIONLabel(Enum):
    CONDITIONS = "conditions_PASSION"
    IMPETIGO = "impetig"
    SKIN_CONDITIONS = "fitzpatrick"


def extract_subject_id(path: str):
    pattern = r"Files_Matrix_([A-Za-z0-9]+)"
    match = re.search(pattern, path)
    if match:
        return str(match.group(1)).strip()
    else:
        return np.nan


class PASSIONDataset(GenericImageDataset):
    """PASSION dataset."""

    IMG_COL = "img_path"
    LBL_COL = None

    def __init__(
        self,
        dataset_dir: Union[str, Path] = "data/PASSION/",
        meta_data_file: Union[str, Path] = "passion_cleaned_final_case_level.csv",
        split_file: Union[str, Path, None] = None,
        transform=None,
        val_transform=None,
        label_col: Union[PASSIONLabel, str] = PASSIONLabel.CONDITIONS,
        return_path: bool = False,
        image_extensions: Sequence = (
            "*.jpeg",
            "*.jpg",
            "*.JPG",
            "*.JPEG",
            "*.PNG",
            "*.png",
        ),
        return_embedding: bool = False,
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
        if isinstance(label_col, str):
            label_col = PASSIONLabel[label_col]
        self.LBL_COL = label_col.value
        super().__init__(
            dataset_dir=dataset_dir,
            transform=transform,
            val_transform=val_transform,
            return_path=return_path,
            image_extensions=image_extensions,
            **kwargs,
        )
        meta_data_file = self.check_path(self.dataset_dir / meta_data_file)
        self.meta_data["subject_id"] = self.meta_data.img_path.apply(extract_subject_id)
        # get the labels of meta-data `PASSION_Files`
        passion_meta_data = pd.read_csv(meta_data_file, index_col=0)
        self.LBL_COL = self.LBL_COL.replace("lbl_", "")
        self.meta_data = self.meta_data.drop(
            columns=[self.LBL_COL, f"lbl_{self.LBL_COL}"], axis=1
        ).merge(passion_meta_data, on=["subject_id"], how="inner")
        # fill the `impetig` column
        impetigo_mapper = {0.0: "not impetiginized", 1.0: "impetiginized"}
        self.meta_data["impetig"] = self.meta_data["impetig"].fillna(value=0.0)
        self.meta_data["impetig"] = self.meta_data["impetig"].apply(impetigo_mapper.get)
        # better naming for the skin type color
        self.meta_data["fitzpatrick"] = self.meta_data["fitzpatrick"].apply(
            lambda fst: f"fitzpatrick skin type {fst}"
        )
        # split the `body_loc` column
        self.meta_data["body_loc"] = self.meta_data["body_loc"].fillna("None")
        self.meta_data["body_loc"] = self.meta_data["body_loc"].str.split(";")
        self.meta_data["body_loc"] = self.meta_data["body_loc"].apply(
            lambda lst: [x.strip().lower() for x in lst]
        )
        # get the splitting type
        if split_file is not None:
            split_file = self.check_path(self.dataset_dir / split_file)
            df_split = pd.read_csv(split_file)
            self.meta_data = self.meta_data.merge(
                df_split, on="subject_id", how="inner"
            )
            self.meta_data.reset_index(drop=True, inplace=True)
            del df_split

        self.meta_data.reset_index(drop=True, inplace=True)
        int_lbl, lbl_mapping = pd.factorize(self.meta_data[self.LBL_COL])
        self.LBL_COL = f"lbl_{self.LBL_COL}"
        self.meta_data[self.LBL_COL] = int_lbl

        self.return_path = return_path
        self.return_embedding = return_embedding
        self.classes = list(lbl_mapping)
        self.n_classes = len(self.classes)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        img_name = self.meta_data.loc[self.meta_data.index[index], self.IMG_COL]
        diagnosis = self.meta_data.loc[self.meta_data.index[index], self.LBL_COL]
        if self.return_embedding:
            embedding = self.meta_data.loc[self.meta_data.index[index], "embedding"]
            embedding = torch.Tensor(embedding)
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
