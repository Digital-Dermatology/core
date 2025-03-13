import os
from pathlib import Path
from typing import Optional, Sequence, Union

import pandas as pd
import torch

from ....src.datasets.base_dataset import BaseDataset
from ....src.datasets.utils import clahe


class VinDrBodyPartXRDataset(BaseDataset):
    """VinDr-BodyPartXR Dataset: https://vindr.ai/datasets/bodypartxr"""

    IMG_COL = "img_path"
    LBL_COL = "diagnosis"

    def __init__(
        self,
        dataset_dir: Union[str, Path] = "data/VinDr-BodyPartXR/",
        transform=None,
        val_transform=None,
        return_path: bool = False,
        load_dicom: bool = False,
        image_extensions: Sequence = ("*.png"),
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
        super().__init__(transform=transform, val_transform=val_transform, **kwargs)
        # check if the dataset path exists
        self.dataset_dir = Path(dataset_dir)
        if not self.dataset_dir.exists():
            raise ValueError(f"Image path must exist, path: {self.dataset_dir}")

        if not (self.dataset_dir / "metadata.csv").exists():
            # this can be used to load the "raw" dataset
            if load_dicom:
                image_extensions = ("*.dcm", "*.dicom")

            # create dicts for retreiving imgs
            l_files = []
            for extension in image_extensions:
                l_files.extend(
                    VinDrBodyPartXRDataset.find_files_with_extension(
                        directory_path=self.dataset_dir,
                        extension=extension,
                    )
                )

            # create the metadata dataframe
            self.meta_data = pd.DataFrame(set(l_files))
            self.meta_data.columns = [self.IMG_COL]
            self.meta_data["img_name"] = self.meta_data[self.IMG_COL].apply(
                lambda x: os.path.splitext(os.path.basename(x))[0]
            )
            self.meta_data[self.LBL_COL] = self.meta_data[self.IMG_COL].apply(
                lambda x: Path(x).parents[0].name
            )
            self.meta_data.to_csv(self.dataset_dir / "metadata.csv", index=False)
        else:
            self.meta_data = pd.read_csv(self.dataset_dir / "metadata.csv")
            self.meta_data["img_path"] = self.meta_data["img_path"].apply(
                lambda x: str(self.dataset_dir / x)
            )

        # remove data quality issues if file is given
        self.remove_data_quality_issues(data_quality_issues_list)
        self.meta_data.reset_index(drop=True, inplace=True)
        int_lbl, lbl_mapping = pd.factorize(self.meta_data[self.LBL_COL])
        self.meta_data["lbl_diagnosis"] = int_lbl

        # global configs
        self.return_path = return_path
        self.classes = list(lbl_mapping)
        self.n_classes = len(self.classes)

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.meta_data.loc[self.meta_data.index[idx], self.IMG_COL]
        image = clahe(img_name)

        if self.transform and self.training:
            image = self.transform(image)
        elif self.val_transform and not self.training:
            image = self.val_transform(image)

        diagnosis = self.meta_data.loc[self.meta_data.index[idx], "lbl_diagnosis"]
        if self.return_path:
            return image, img_name, int(diagnosis)
        else:
            return image, int(diagnosis)
