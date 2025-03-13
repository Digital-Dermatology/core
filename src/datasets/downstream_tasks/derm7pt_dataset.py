import re
from enum import Enum
from pathlib import Path
from typing import Optional, Sequence, Union

import pandas as pd

from ....src.datasets.generic_image_dataset import GenericImageDataset


class Derm7ptLabel(Enum):
    DISEASE = "diagnosis"


class Derm7ptDataset(GenericImageDataset):
    """Derm7pt dataset."""

    IMG_COL = "img_path"
    LBL_COL = "diagnosis"

    def __init__(
        self,
        dataset_dir: Union[str, Path] = "data/dataset/",
        csv_file: Union[str, Path] = "data/derm7pt/meta/meta.csv",
        transform=None,
        val_transform=None,
        label_col: Derm7ptLabel = Derm7ptLabel.DISEASE,
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
        csv_file = self.check_path(csv_file)
        # load the metadata
        df_meta = pd.DataFrame(pd.read_csv(csv_file))
        # pre-process the dataframe
        df_meta["diagnosis"] = df_meta["diagnosis"].apply(
            lambda x: re.sub(r"\(.*\)", "", x).strip()
        )
        df_meta["lbl_diagnosis"] = pd.factorize(df_meta["diagnosis"])[0]
        df_meta = df_meta[["clinic", "derm", "diagnosis", "lbl_diagnosis"]]

        _df_clinic = pd.DataFrame(df_meta[["clinic", "diagnosis", "lbl_diagnosis"]])
        _df_clinic.rename(columns={"clinic": "img_path"}, inplace=True)
        _df_derm = pd.DataFrame(df_meta[["derm", "diagnosis", "lbl_diagnosis"]])
        _df_derm.rename(columns={"derm": "img_path"}, inplace=True)

        df_meta = pd.concat([_df_clinic, _df_derm])
        df_meta["img_path"] = df_meta["img_path"].apply(
            lambda x: f"{str(dataset_dir)}/{x}"
        )
        df_meta["img_name"] = df_meta["img_path"].apply(lambda x: Path(x).stem)
        df_meta.reset_index(drop=True, inplace=True)
        self.meta_data = df_meta

        # remove data quality issues if file is given
        self.remove_data_quality_issues(data_quality_issues_list)
        self.meta_data.reset_index(drop=True, inplace=True)

        # Global configs
        self.LBL_COL = f"lbl_{label_col.value}"
        self.return_path = return_path
        self.classes = (
            self.meta_data["diagnosis"].unique().tolist()
            if label_col == Derm7ptLabel.DISEASE
            else ["benign", "malignant"]
        )
        self.n_classes = len(self.classes)
