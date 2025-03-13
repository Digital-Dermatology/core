from pathlib import Path
from typing import Sequence, Union

import pandas as pd

from ....src.datasets.generic_image_dataset import GenericImageDataset


class ISIC2019Dataset(GenericImageDataset):
    """ISIC 2019 image dataset."""

    IMG_COL = "img_path"
    LBL_COL = "diagnosis"

    def __init__(
        self,
        dataset_dir: Union[str, Path] = "data/dataset/",
        isic_train_gt_name: Union[str, Path] = "ISIC_2019_Training_GroundTruth.csv",
        isic_train_meta_name: Union[str, Path] = "ISIC_2019_Training_Metadata.csv",
        isic_test_meta_name: Union[str, Path] = "ISIC_2019_Test_Metadata.csv",
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
        # add label info
        labels = pd.read_csv(self.dataset_dir / isic_train_gt_name)
        self.meta_data = self.meta_data.merge(
            labels,
            left_on="img_name",
            right_on="image",
            how="outer",
        )
        self.meta_data.drop(columns=["image"], inplace=True)
        # add meta info
        meta_train = pd.read_csv(self.dataset_dir / isic_train_meta_name)
        meta_test = pd.read_csv(self.dataset_dir / isic_test_meta_name)
        meta = pd.concat([meta_train, meta_test])
        self.meta_data = self.meta_data.merge(
            meta,
            left_on="img_name",
            right_on="image",
            how="inner",
        )
        self.meta_data.drop(columns=["image"], inplace=True)

        # parse one hot label column
        self.meta_data["dataset_split"] = self.meta_data["diagnosis"]
        self.meta_data["diagnosis"] = self.meta_data[
            ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]
        ].idxmax(axis=1)
        self.meta_data["diagnosis"].fillna("UNK", inplace=True)
        int_lbl, lbl_mapping = pd.factorize(self.meta_data["diagnosis"])
        self.meta_data["lbl_diagnosis"] = int_lbl

        # Global configs
        self.classes = list(lbl_mapping)
        self.n_classes = len(self.classes)
