import json
from pathlib import Path
from typing import Optional, Sequence, Union

import pandas as pd

from ....src.datasets.generic_image_dataset import GenericImageDataset


class OxfordFlower102Dataset(GenericImageDataset):
    """Oxford Flower 102 image dataset."""

    def __init__(
        self,
        dataset_dir: Union[str, Path] = "data/dataset/",
        class_mapper_file_name: Union[str, Path] = "cat_to_name.json",
        transform=None,
        val_transform=None,
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
        # parse the data split from the path
        self.meta_data["data_split"] = self.meta_data["img_path"].apply(
            lambda x: Path(x).parts[-3]
        )

        # remove data quality issues if file is given
        self.remove_data_quality_issues(data_quality_issues_list)
        self.meta_data.reset_index(drop=True, inplace=True)

        # get the class mapper
        with open(self.dataset_dir / class_mapper_file_name) as f:
            class_mapper = json.load(f)
        self.meta_data["diagnosis"] = self.meta_data["diagnosis"].apply(
            lambda x: class_mapper.get(x, "Unknown")
        )
        int_lbl, lbl_mapping = pd.factorize(self.meta_data["diagnosis"])
        self.meta_data[self.LBL_COL] = int_lbl

        # Global configs
        self.classes = list(lbl_mapping)
        self.n_classes = len(self.classes)
