import os
from pathlib import Path
from typing import Optional, Sequence, Union

import pandas as pd
import torch
from PIL import Image

from ....src.datasets.base_dataset import BaseDataset


class DaffodilDataset(BaseDataset):
    """Daffodil dataset - organized by condition subfolders."""

    IMG_COL = "img_path"
    LBL_COL = "condition"

    def __init__(
        self,
        dataset_dir: Union[str, Path] = "/media/onix/data-repository/Daffodil",
        transform=None,
        val_transform=None,
        return_path: bool = False,
        image_extensions: Sequence = ("*.png", "*.jpg", "*.jpeg", "*.JPEG", "*.JPG"),
        data_quality_issues_list: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        """
        Initializes the Daffodil dataset.

        Parameters
        ----------
        dataset_dir : str
            Directory with condition subfolders containing images.
        transform : Union[callable, optional]
            Optional transform to be applied to the images.
        val_transform : Union[callable, optional]
            Optional transform to be applied to the images when in validation mode.
        return_path : bool
            If the path of the image should be returned or not.
        image_extensions : Sequence
            Image file extensions to search for.
        """
        super().__init__(transform=transform, val_transform=val_transform, **kwargs)

        # Check if the dataset path exists
        self.dataset_dir = Path(dataset_dir)
        if not self.dataset_dir.exists():
            raise ValueError(f"Dataset path must exist, path: {self.dataset_dir}")

        # Find all image files in subdirectories
        l_files = []
        for extension in image_extensions:
            l_files.extend(
                DaffodilDataset.find_files_with_extension(
                    directory_path=dataset_dir,
                    extension=extension,
                )
            )

        # Create the metadata dataframe
        self.meta_data = pd.DataFrame(list(set(l_files)))
        self.meta_data.columns = [self.IMG_COL]

        # Extract condition from parent directory name
        self.meta_data[self.LBL_COL] = self.meta_data[self.IMG_COL].apply(
            lambda x: Path(x).parent.name
        )

        # Map condition folder names to readable descriptions
        condition_map = {
            "acne": "acne",
            "hyperpigmentation": "hyperpigmentation",
            "Nail_psoriasis": "nail psoriasis",
            "SJS-TEN": "Stevens-Johnson syndrome or toxic epidermal necrolysis",
            "Vitiligo": "vitiligo",
        }

        self.meta_data[self.LBL_COL] = self.meta_data[self.LBL_COL].map(
            lambda x: condition_map.get(x, x)
        )

        # Create descriptions
        self.meta_data["description"] = self.meta_data.apply(
            lambda row: f"A clinical image showing {row[self.LBL_COL]}.",
            axis=1,
        )

        # Create dataset_desc column
        self.meta_data["dataset_desc"] = "Daffodil"

        # Create image names
        self.meta_data["img_name"] = self.meta_data[self.IMG_COL].apply(
            lambda x: os.path.splitext(os.path.basename(x))[0]
        )

        # Filter out images that don't exist
        existing_mask = self.meta_data[self.IMG_COL].apply(lambda x: Path(x).exists())
        self.meta_data = self.meta_data[existing_mask].copy()
        self.meta_data.reset_index(drop=True, inplace=True)

        # Remove data quality issues if file is given
        self.remove_data_quality_issues(data_quality_issues_list)
        self.meta_data.reset_index(drop=True, inplace=True)

        # Global configs
        self.return_path = return_path
        self.classes = list(self.meta_data[self.LBL_COL].unique())
        self.n_classes = len(self.classes)

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

        description = self.meta_data.loc[self.meta_data.index[idx], "description"]
        if self.return_path:
            return image, img_name, description
        else:
            return image, description
