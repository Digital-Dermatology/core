import re
from enum import Enum
from pathlib import Path
from typing import Optional, Sequence, Union

import pandas as pd

from ....src.datasets.generic_image_dataset import GenericImageDataset


class SD128Label(Enum):
    DISEASE = "diagnosis"


class SD128Dataset(GenericImageDataset):
    """SD-128 dataset."""

    IMG_COL = "img_path"
    LBL_COL = "diagnosis"

    def __init__(
        self,
        dataset_dir: Union[str, Path] = "data/dataset/",
        transform=None,
        val_transform=None,
        label_col: SD128Label = SD128Label.DISEASE,
        return_path: bool = False,
        image_extensions: Sequence = ("*.png", "*.jpg", "*.JPEG"),
        data_quality_issues_list: Optional[Union[str, Path]] = None,
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
        super().__init__(
            dataset_dir=dataset_dir,
            transform=transform,
            val_transform=val_transform,
            return_path=return_path,
            image_extensions=image_extensions,
            **kwargs,
        )
        # post-process metadata
        meta_data = self.meta_data
        meta_data["diagnosis"] = meta_data["diagnosis"].apply(
            lambda x: re.sub(r"\(.*\)", "", x).strip()
        )
        meta_data["lbl_diagnosis"] = pd.factorize(meta_data["diagnosis"])[0]
        meta_data.reset_index(drop=True, inplace=True)
        self.meta_data = meta_data

        # remove data quality issues if file is given
        self.remove_data_quality_issues(data_quality_issues_list)
        self.meta_data.reset_index(drop=True, inplace=True)

        # Global configs
        self.LBL_COL = f"lbl_{label_col.value}"
        self.return_path = return_path
        self.classes = (
            self.meta_data["diagnosis"].unique().tolist()
            if label_col == SD128Label.DISEASE
            else ["benign", "malignant"]
        )
        self.n_classes = len(self.classes)
