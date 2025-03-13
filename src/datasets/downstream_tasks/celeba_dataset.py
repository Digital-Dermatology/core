from pathlib import Path
from typing import Sequence, Union

import pandas as pd

from ....src.datasets.generic_image_dataset import GenericImageDataset


class CelebADataset(GenericImageDataset):
    """CelebA image dataset."""

    IMG_COL = "img_path"
    LBL_COL = "diagnosis"

    def __init__(
        self,
        dataset_dir: Union[str, Path] = "data/dataset/",
        identity_mapping: Union[str, Path] = "../../Anno/identity_CelebA.txt",
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
        # Merge with identity mapping
        identity_mapping = self.check_path(identity_mapping)
        df_meta = pd.read_csv(identity_mapping, sep=" ", header=None)
        df_meta.columns = ["img_name", "celeb_id"]
        df_meta["img_name"] = df_meta["img_name"].apply(lambda x: x.split(".")[0])
        self.meta_data = self.meta_data.merge(df_meta, on="img_name", how="left")
        int_lbl, lbl_mapping = pd.factorize(self.meta_data["celeb_id"])
        self.meta_data["lbl_diagnosis"] = int_lbl

        # Global configs
        self.classes = list(lbl_mapping)
        self.n_classes = len(self.classes)
