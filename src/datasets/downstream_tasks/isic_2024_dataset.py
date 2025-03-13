from pathlib import Path
from typing import Sequence, Union

import pandas as pd

from ....src.datasets.generic_image_dataset import GenericImageDataset


class ISIC2024Dataset(GenericImageDataset):
    """ISIC 2024 image dataset."""

    IMG_COL = "img_path"
    LBL_COL = "diagnosis"

    def __init__(
        self,
        dataset_dir: Union[str, Path] = "data/dataset/",
        isic_train_meta_name: Union[str, Path] = "train-metadata.csv",
        transform=None,
        val_transform=None,
        return_path: bool = False,
        image_extensions: Sequence = ("*.png", "*.jpg", "*.JPEG"),
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
        df_meta = pd.read_csv(self.dataset_dir / isic_train_meta_name)
        self.meta_data = self.meta_data.merge(
            df_meta,
            left_on="img_name",
            right_on="isic_id",
            how="outer",
        )

        class_mapper = {0: "benign", 1: "malignant"}
        self.meta_data["target"] = self.meta_data["target"].apply(
            lambda x: class_mapper.get(x)
        )
        self.meta_data["diagnosis"] = self.meta_data["target"]
        int_lbl, lbl_mapping = pd.factorize(self.meta_data["diagnosis"])
        self.meta_data["lbl_diagnosis"] = int_lbl

        # Global configs
        self.classes = list(lbl_mapping)
        self.n_classes = len(self.classes)
