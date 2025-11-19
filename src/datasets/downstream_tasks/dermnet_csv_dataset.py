from pathlib import Path
from typing import Sequence, Union

import pandas as pd

from ....src.datasets.generic_image_dataset import GenericImageDataset


class DermNetCSVDataset(GenericImageDataset):
    """DermNet image dataset with CSV metadata."""

    IMG_COL = "img_path"
    LBL_COL = "diagnosis"

    def __init__(
        self,
        dataset_dir: Union[str, Path] = "data/dataset/",
        dermnet_meta_name: Union[str, Path] = "dermnet.csv",
        transform=None,
        val_transform=None,
        return_path: bool = False,
        image_extensions: Sequence = ("*.jpg", "*.JPG", "*.png", "*.JPEG"),
        **kwargs,
    ):
        """
        Initializes the DermNet CSV dataset.

        Parameters
        ----------
        dataset_dir : str
            Directory with all the images.
        dermnet_meta_name : str
            Name of the metadata CSV file.
        transform : Union[callable, optional]
            Optional transform to be applied to the images.
        val_transform : Union[callable, optional]
            Optional transform to be applied to the images when in validation mode.
        return_path : bool
            If the path of the image should be returned or not.
        image_extensions : Sequence
            Sequence of image file extensions to search for.
        """
        super().__init__(
            dataset_dir=dataset_dir,
            transform=transform,
            val_transform=val_transform,
            return_path=return_path,
            image_extensions=image_extensions,
            **kwargs,
        )

        # Load metadata CSV directly
        self.meta_data = pd.read_csv(self.dataset_dir / dermnet_meta_name)

        # Create img_path from image column
        self.meta_data["img_path"] = self.meta_data["image"].apply(
            lambda x: str(self.dataset_dir / "images" / x)
        )

        self.meta_data = self.meta_data[self.meta_data["img_path"].notna()]
        self.meta_data.reset_index(drop=True, inplace=True)

        int_lbl, lbl_mapping = pd.factorize(self.meta_data["diagnosis"])
        self.meta_data["lbl_diagnosis"] = int_lbl

        def create_description(row):
            """Create a clinical description for the image."""
            diagnosis = row.get("diagnosis")
            if pd.notna(diagnosis):
                return f"A clinical image showing {diagnosis.lower()}."
            return "A clinical dermatology image."

        self.meta_data["description"] = self.meta_data.apply(create_description, axis=1)
        self.meta_data = self.meta_data.rename(
            columns={
                "diagnosis": "condition",
            },
        )
        self.meta_data["dataset_desc"] = "DermNet-CSV"

        # Global configs
        self.classes = list(lbl_mapping)
        self.n_classes = len(self.classes)
