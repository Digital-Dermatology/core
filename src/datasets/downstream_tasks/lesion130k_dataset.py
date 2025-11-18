import os
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import torch
from PIL import Image

from ....src.datasets.base_dataset import BaseDataset


class LESION130kDataset(BaseDataset):
    """LESION130k dataset - skin lesion detection dataset."""

    IMG_COL = "img_path"
    LBL_COL = "description"

    def __init__(
        self,
        dataset_dir: Union[str, Path] = "/media/onix/data-repository/LESION130k",
        csv_file: Union[str, Path] = None,
        transform=None,
        val_transform=None,
        return_path: bool = False,
        data_quality_issues_list: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        """
        Initializes the LESION130k dataset.

        Parameters
        ----------
        dataset_dir : str
            Directory with images subfolder and all.csv metadata file.
        csv_file : str
            Path to the CSV metadata file. If None, uses default location.
        transform : Union[callable, optional]
            Optional transform to be applied to the images.
        val_transform : Union[callable, optional]
            Optional transform to be applied to the images when in validation mode.
        return_path : bool
            If the path of the image should be returned or not.
        """
        super().__init__(transform=transform, val_transform=val_transform, **kwargs)

        # Check if the dataset path exists
        self.dataset_dir = Path(dataset_dir)
        if not self.dataset_dir.exists():
            raise ValueError(f"Dataset path must exist, path: {self.dataset_dir}")

        # Set CSV file path
        if csv_file is None:
            csv_file = self.dataset_dir / "images" / "all.csv"
        else:
            csv_file = Path(csv_file)

        if not csv_file.exists():
            raise ValueError(f"CSV file must exist, path: {csv_file}")

        # Load metadata
        # The CSV has columns: ##, no label, x_center, y_center, width, height, url
        # We'll skip the first row and read it properly
        self.meta_data = pd.read_csv(csv_file, skiprows=1, sep=" ")

        # The first column is the image ID (e.g., "00000")
        self.meta_data.columns = [
            "image_id",
            "dataset_source",
            "x_center",
            "y_center",
            "width",
            "height",
            "url",
        ]

        # Construct full image paths
        # Images are stored as {image_id}.jpg in the images folder
        self.meta_data[self.IMG_COL] = self.meta_data["image_id"].apply(
            lambda x: str(self.dataset_dir / "images" / f"{x}.jpg")
        )

        # Create descriptions - generic for lesion detection
        self.meta_data["description"] = "A clinical photograph of a skin lesion."

        # Create dataset_desc column
        self.meta_data["dataset_desc"] = "LESION130k"

        # Create image names
        self.meta_data["img_name"] = self.meta_data["image_id"]

        # Add condition field - we don't have specific condition info, so use generic
        self.meta_data["condition"] = "skin lesion"

        # Filter out images that don't exist
        existing_mask = self.meta_data[self.IMG_COL].apply(lambda x: Path(x).exists())
        self.meta_data = self.meta_data[existing_mask].copy()
        self.meta_data.reset_index(drop=True, inplace=True)

        # Remove data quality issues if file is given
        self.remove_data_quality_issues(data_quality_issues_list)
        self.meta_data.reset_index(drop=True, inplace=True)

        # Global configs
        self.return_path = return_path
        self.classes = ["skin_lesion"]
        self.n_classes = 1

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
