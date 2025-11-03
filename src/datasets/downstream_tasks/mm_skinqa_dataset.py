import os
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import torch
from PIL import Image

from ....src.datasets.base_dataset import BaseDataset


class MMSkinQADataset(BaseDataset):
    """MM-SkinQA dataset - multimodal skin QA dataset with captions."""

    IMG_COL = "img_path"
    LBL_COL = "description"

    def __init__(
        self,
        dataset_dir: Union[str, Path] = "/media/onix/data-repository/MM-SkinQA",
        csv_file: Union[str, Path] = None,
        transform=None,
        val_transform=None,
        return_path: bool = False,
        data_quality_issues_list: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        """
        Initializes the MM-SkinQA dataset.

        Parameters
        ----------
        dataset_dir : str
            Directory with image files and caption.csv metadata file.
        csv_file : str
            Path to the CSV metadata file. If None, uses caption.csv.
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
            csv_file = self.dataset_dir / "caption.csv"
        else:
            csv_file = Path(csv_file)

        if not csv_file.exists():
            raise ValueError(f"CSV file must exist, path: {csv_file}")

        # Load metadata
        # Columns: image, caption, modality, sex, age, cleaned_caption
        self.meta_data = pd.read_csv(csv_file)

        # Construct full image paths
        # Images are in dataset/ subfolder relative to CSV location
        self.meta_data[self.IMG_COL] = self.meta_data["image"].apply(
            lambda x: str(self.dataset_dir / x) if not os.path.isabs(x) else x
        )

        # Use cleaned_caption if available, otherwise use caption
        if "cleaned_caption" in self.meta_data.columns:
            self.meta_data["description"] = self.meta_data["cleaned_caption"].fillna(
                self.meta_data["caption"]
            )
        else:
            self.meta_data["description"] = self.meta_data["caption"]

        # Clean up descriptions - ensure they're strings and handle NaN
        self.meta_data["description"] = self.meta_data["description"].apply(
            lambda x: str(x) if pd.notna(x) else "A dermatological image."
        )

        # Create dataset_desc column
        self.meta_data["dataset_desc"] = "MM-SkinQA"

        # Create image names
        self.meta_data["img_name"] = self.meta_data["image"].apply(
            lambda x: os.path.splitext(os.path.basename(x))[0]
        )

        # Add available metadata columns
        # Rename 'sex' to 'gender' for consistency with other datasets
        if "sex" in self.meta_data.columns:
            self.meta_data["gender"] = self.meta_data["sex"]

        # Keep modality and age if available
        # Age is already in the correct format

        # Filter out images that don't exist
        existing_mask = self.meta_data[self.IMG_COL].apply(lambda x: Path(x).exists())
        self.meta_data = self.meta_data[existing_mask].copy()
        self.meta_data.reset_index(drop=True, inplace=True)

        # Remove data quality issues if file is given
        self.remove_data_quality_issues(data_quality_issues_list)
        self.meta_data.reset_index(drop=True, inplace=True)

        # Global configs
        self.return_path = return_path
        self.classes = ["dermatology_image"]
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
