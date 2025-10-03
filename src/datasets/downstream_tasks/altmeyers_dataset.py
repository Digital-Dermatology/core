from pathlib import Path
from typing import Optional, Union

import pandas as pd
import torch
from PIL import Image

from ....src.datasets.base_dataset import BaseDataset


class AltmeyersDataset(BaseDataset):
    """Altmeyers dermatology atlas dataset."""

    IMG_COL = "img_path"
    LBL_COL = "description"

    def __init__(
        self,
        dataset_dir: Union[str, Path] = "/media/onix/data-repository/Altmeyers",
        transform=None,
        val_transform=None,
        return_path: bool = False,
        data_quality_issues_list: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        """
        Initializes the dataset.

        Parameters
        ----------
        dataset_dir : str
            Directory with all the images and merged.csv file.
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

        # Load metadata CSV
        metadata_path = self.dataset_dir / "merged.csv"
        if not metadata_path.exists():
            raise ValueError(f"Metadata file must exist, path: {metadata_path}")

        self.meta_data = pd.read_csv(metadata_path)

        # Create full image paths (images have no extension, they are JPEG files)
        self.meta_data[self.IMG_COL] = self.meta_data["path"].apply(
            lambda x: str(self.dataset_dir / Path(x).name)
        )

        # Use caption as description with light preprocessing
        self.meta_data["description"] = self.meta_data["caption"].apply(self._preprocess_caption)

        # Create image names
        self.meta_data["img_name"] = self.meta_data["path"].apply(
            lambda x: Path(x).stem
        )

        # Add body location if available
        if "localization" in self.meta_data.columns:
            self.meta_data["body_location"] = self.meta_data["localization"]

        # Filter out images that don't exist
        existing_mask = self.meta_data[self.IMG_COL].apply(lambda x: Path(x).exists())
        self.meta_data = self.meta_data[existing_mask].copy()
        self.meta_data.reset_index(drop=True, inplace=True)

        # Remove data quality issues if file is given
        self.remove_data_quality_issues(data_quality_issues_list)
        self.meta_data.reset_index(drop=True, inplace=True)

        # Global configs
        self.return_path = return_path
        self.classes = ["dermatology_image"]  # Generic class
        self.n_classes = 1

    def _preprocess_caption(self, caption):
        """Light preprocessing of captions for consistency."""
        import re

        if pd.isna(caption) or not caption:
            return "Dermatology image."

        caption = str(caption).strip()

        # Remove excessive whitespace and newlines
        caption = re.sub(r'\s+', ' ', caption)
        caption = re.sub(r'\n+', ' ', caption)

        # Remove common German/English prefixes with condition names
        # Pattern: "Condition name: description" -> keep both
        # We don't remove the condition name as it's valuable information

        # Clean up excessive punctuation
        caption = re.sub(r'[.]{2,}', '.', caption)
        caption = re.sub(r'[,]{2,}', ',', caption)
        caption = re.sub(r'[\s]+([.,;:])', r'\1', caption)  # Remove space before punctuation

        # Ensure proper sentence structure
        caption = caption.strip()
        if caption and not caption.endswith(('.', '!', '?')):
            caption += '.'

        # Capitalize first letter
        if caption:
            caption = caption[0].upper() + caption[1:]

        return caption if caption else "Dermatology image."

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
