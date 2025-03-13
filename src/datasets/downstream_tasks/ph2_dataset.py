from pathlib import Path
from typing import Optional, Sequence, Union

import pandas as pd

from ....src.datasets.generic_image_dataset import GenericImageDataset


class PH2Dataset(GenericImageDataset):
    """PH2 dataset."""

    IMG_COL = "img_path"
    LBL_COL = "diagnosis"

    def __init__(
        self,
        dataset_dir: Union[str, Path] = "data/dataset/",
        csv_file: Union[str, Path] = "data/PH2Dataset/PH2_dataset.xlsx",
        transform=None,
        val_transform=None,
        return_path: bool = False,
        image_extensions: Sequence = ("*.bmp"),
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
        csv_file = self.check_path(csv_file)
        # load the metadata
        df_meta = pd.DataFrame(pd.read_excel(csv_file, header=12))
        # pre-process the dataframe
        label_cols = ["Common Nevus", "Atypical Nevus", "Melanoma"]
        df_meta[label_cols] = df_meta[label_cols].fillna(0)
        df_meta[label_cols] = df_meta[label_cols].replace("X", 1)
        df_meta["Ohot_lbl"] = df_meta[label_cols].values.argmax(1)
        # encode the label
        class_mapper = {0: "Common Nevus", 1: "Atypical Nevus", 2: "Melanoma"}
        df_meta["label"] = df_meta["Ohot_lbl"].apply(class_mapper.get)
        # merge with existing metadata
        df_meta = self.meta_data.merge(
            df_meta[["Image Name", "Ohot_lbl", "label"]],
            left_on="img_name",
            right_on="Image Name",
        )
        df_meta.drop(columns=["lbl_diagnosis", "diagnosis"], inplace=True)
        df_meta.rename(
            columns={"Ohot_lbl": "lbl_diagnosis", "label": "diagnosis"}, inplace=True
        )
        self.meta_data = df_meta
        del df_meta

        # remove data quality issues if file is given
        self.remove_data_quality_issues(data_quality_issues_list)
        self.meta_data.reset_index(drop=True, inplace=True)

        # Global configs
        self.classes = label_cols
        self.n_classes = len(self.classes)
