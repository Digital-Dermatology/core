import os
import re
from enum import Enum
from pathlib import Path
from typing import Sequence, Union

import pandas as pd
import torch

from ....src.datasets.base_dataset import BaseDataset
from ....src.datasets.utils import clahe


class CheXpertLabel(Enum):
    # only the competition labels
    # col name of label code, col name of label
    ATELECTASIS = "Atelectasis_lbl", "Atelectasis"
    CARDIOMEGALY = "Cardiomegaly_lbl", "Cardiomegaly"
    CONSOLIDATION = "Consolidation_lbl", "Consolidation"
    EDEMA = "Edema_lbl", "Edema"
    PLEURAL_EFFUSION = "Pleural_Effusion_lbl", "Pleural Effusion"


class CheXpertDataset(BaseDataset):
    """CheXpert dataset."""

    IMG_COL = "img_path"
    LBL_COL = "diagnosis"

    def __init__(
        self,
        dataset_dir: Union[str, Path] = "data/dataset/",
        images_path_name: str = "CheXpert-v1.0-Resized-512",
        train_meta_name: str = "train.csv",
        val_meta_name: str = "valid.csv",
        test_meta_name: str = "test_labels.csv",
        label_col: CheXpertLabel = CheXpertLabel.ATELECTASIS,
        transform=None,
        val_transform=None,
        return_path: bool = False,
        image_extensions: Sequence = ("*.jpg"),
        high_quality: bool = False,
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

        # create dicts for retreiving imgs
        l_files = []
        for extension in image_extensions:
            l_files.extend(
                CheXpertDataset.find_files_with_extension(
                    directory_path=self.dataset_dir / images_path_name,
                    extension=extension,
                )
            )

        if high_quality:
            # only consider validation and test images
            # as they have >3 experts
            hq_sets = ["valid", "test"]
            l_files = [
                x
                for x in l_files
                if x.replace(str(self.dataset_dir / images_path_name), "").split("/")[1]
                in hq_sets
            ]

        # create the metadata dataframe
        self.meta_data = pd.DataFrame(set(l_files))
        self.meta_data.columns = [self.IMG_COL]
        self.meta_data["img_name"] = self.meta_data[self.IMG_COL].apply(
            lambda x: os.path.splitext(os.path.basename(x))[0]
        )
        self.meta_data[self.LBL_COL] = self.meta_data[self.IMG_COL].apply(
            lambda x: Path(x).parents[0].name
        )
        self.meta_data = self.meta_data[
            self.meta_data["img_path"].apply(lambda x: ".jpg" in x)
        ]
        self.meta_data.reset_index(drop=True, inplace=True)

        self.meta_data["patient_id"] = self.meta_data["img_path"].apply(
            lambda x: re.findall(r"patient\d*", str(x))[0]
        )
        self.meta_data["relative_img_path"] = self.meta_data["img_path"].apply(
            lambda x: x.replace(f"{str(self.dataset_dir)}/", "").replace(
                images_path_name, "CheXpert-v1.0"
            )
        )
        # merge with meta data
        meta_train = pd.read_csv(self.dataset_dir / images_path_name / train_meta_name)
        meta_val = pd.read_csv(self.dataset_dir / images_path_name / val_meta_name)
        meta_test = pd.read_csv(self.dataset_dir / images_path_name / test_meta_name)
        meta_test["Path"] = "CheXpert-v1.0/" + meta_test["Path"]
        meta_train["dataset_origin"] = "Train"
        meta_val["dataset_origin"] = "Validation"
        meta_test["dataset_origin"] = "Test"
        meta_df = pd.concat([meta_train, meta_val, meta_test])
        meta_df.reset_index(drop=True, inplace=True)
        self.meta_data = self.meta_data.merge(
            meta_df,
            left_on="relative_img_path",
            right_on="Path",
            how="inner",
        )
        del meta_train, meta_val, meta_df
        # merge with demographic data
        demo_df = pd.read_excel(self.dataset_dir / "CHEXPERT DEMO.xlsx")
        self.meta_data = self.meta_data.merge(
            demo_df,
            left_on="patient_id",
            right_on="PATIENT",
            how="left",
        )
        del demo_df

        if high_quality:
            # only take one scan per patient
            self.meta_data = self.meta_data.drop_duplicates(subset=["patient_id"])
            self.meta_data.reset_index(drop=True, inplace=True)

        # process the label column
        label_parser = {
            1.0: "positive",
            -1.0: "negative",
            0.0: "uncertain",
        }
        self.meta_data[label_col.value[1]] = self.meta_data[label_col.value[1]].apply(
            lambda x: label_parser.get(x, "not given")
        )
        self.meta_data[label_col.value[0]] = pd.factorize(
            self.meta_data[label_col.value[1]]
        )[0]

        # global configs
        self.return_path = return_path
        self.LBL_COL = label_col.value[0]
        self.classes = self.meta_data[label_col.value[1]].unique().tolist()
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

        diagnosis = self.meta_data.loc[self.meta_data.index[idx], self.LBL_COL]
        if self.return_path:
            return image, img_name, int(diagnosis)
        else:
            return image, int(diagnosis)
