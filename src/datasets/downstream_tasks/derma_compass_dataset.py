import os
from enum import Enum
from glob import glob
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch
from PIL import Image

from ....src.datasets.base_dataset import BaseDataset


class DermaCompassLabel(Enum):
    # col name of label code, col name of label
    PRIMARY = "lbl_primary", "primary"
    SECONDARY = "lbl_secondary", "secondary"


class DermaCompassDataset(BaseDataset):
    """DermaCompass dataset."""

    IMG_COL = "title"
    LBL_COL = "label"

    def __init__(
        self,
        csv_file: Union[str, Path] = "data/DermaCompass/data.csv",
        dataset_dir: Union[str, Path] = "data/DermaCompass/",
        transform=None,
        val_transform=None,
        label_col: DermaCompassLabel = DermaCompassLabel.PRIMARY,
        return_path: bool = False,
        **kwargs,
    ):
        """
        Initializes the dataset.

        Sets the correct path for the needed arguments.

        Parameters
        ----------
        csv_file : str
            Path to the csv file with metadata, including annotations.
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
        self.csv_file = Path(csv_file)
        if not self.csv_file.exists():
            raise ValueError(f"CSV metadata path must exist, path: {self.csv_file}")
        self.dataset_dir = Path(dataset_dir)
        if not self.dataset_dir.exists():
            raise ValueError(f"Image path must exist, path: {self.dataset_dir}")
        # transform the dataframe for better loading
        imageid_path_dict = {
            os.path.splitext(os.path.basename(x))[0]: x
            for x in glob(os.path.join(self.dataset_dir, "*", "*.jpg"))
        }
        # load the metadata
        self.meta_data = pd.DataFrame(pd.read_csv(csv_file, index_col=0))
        self.meta_data["path"] = self.meta_data[self.IMG_COL].map(imageid_path_dict.get)
        self.meta_data = self.meta_data.dropna(subset=["path"])
        self.meta_data = self.meta_data.drop_duplicates(subset="title")
        self.meta_data[DermaCompassLabel.PRIMARY.value[1]] = self.meta_data[
            DermaCompassLabel.PRIMARY.value[1]
        ].replace(np.nan, "undefined")
        self.meta_data[DermaCompassLabel.SECONDARY.value[1]] = self.meta_data[
            DermaCompassLabel.SECONDARY.value[1]
        ].replace(np.nan, "undefined")
        # transform the string labels into categorical values
        self.meta_data[DermaCompassLabel.PRIMARY.value[0]] = pd.factorize(
            self.meta_data[DermaCompassLabel.PRIMARY.value[1]]
        )[0]
        self.meta_data[DermaCompassLabel.SECONDARY.value[0]] = pd.factorize(
            self.meta_data[DermaCompassLabel.SECONDARY.value[1]]
        )[0]
        # create the description
        self.meta_data["description"] = self.meta_data.apply(
            DermaCompassDataset.make_description, axis=1
        )
        self.meta_data = self.meta_data.rename(
            columns={
                "relatedDiseases": "condition",
                "localization": "body_location",
            },
        )
        # global configs
        self.return_path = return_path
        self.IMG_COL = "path"
        self.LBL_COL = label_col.value[0]
        self.classes = self.meta_data[label_col.value[1]].unique().tolist()
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

        diagnosis = self.meta_data.loc[self.meta_data.index[idx], self.LBL_COL]
        if self.return_path:
            return image, img_name, int(diagnosis)
        else:
            return image, int(diagnosis)

    @staticmethod
    def make_description(row):
        parts = []

        # 1) Disease + appearance
        if pd.notna(row.get("relatedDiseases")) or pd.notna(row.get("appearances")):
            rd = row.get("relatedDiseases", "")
            ap = row.get("appearances", "")
            if pd.notna(rd) and pd.notna(ap):
                parts.append(f"{rd} present as {ap} lesions")
            elif pd.notna(rd):
                parts.append(f"{rd} present")
            else:
                parts.append(f"{ap.capitalize()} lesions")

        # 2) Localization & laterality/symmetry
        loc = row.get("localization")
        lat = row.get("localization-lateral")
        sym = row.get("localization-symmetrical")
        exp = row.get("localization-exposure")
        loc_clauses = []
        if pd.notna(lat):
            loc_clauses.append(lat)
        if pd.notna(sym):
            loc_clauses.append(sym)
        if loc_clauses or pd.notna(loc) or pd.notna(exp):
            sub = []
            if loc_clauses:
                sub.append(" and ".join(loc_clauses) + " lateralization/symmetry")
            if pd.notna(loc):
                sub.append(f"on the {loc}")
            if pd.notna(exp):
                sub.append(f"in {exp} areas")
            parts.append(" with " + ", ".join(sub).lstrip(" with "))

        # 3) Distribution & arrangement
        dist = row.get("distribution")
        arr = row.get("arrangement")
        area = row.get("areas")
        dist_clause = []
        if pd.notna(dist):
            dist_clause.append(f"{dist} distributed")
        if pd.notna(arr):
            dist_clause.append(f"a(n) {arr} arrangement")
        if dist_clause:
            clause = " and ".join(dist_clause)
            if pd.notna(area):
                clause += f" across the {area}"
            parts.append(clause)

        # 4) Size
        size = row.get("size-of-single-lesion")
        if pd.notna(size):
            parts.append(f"Individual lesions measure about {size}")

        # 5) Morphology features
        feats = []
        for col, label in [
            ("demarcation", "well-defined borders"),
            ("configuration", "configuration"),
            ("colour", "colouration"),
            ("surface", "surface texture"),
            ("consistency", "consistency"),
            ("structure", "structure"),
        ]:
            val = row.get(col)
            if pd.notna(val):
                # if your val already implies the adjective, you can adjust this
                feats.append(f"{val} {label}")
        if feats:
            parts.append("They have " + ", ".join(feats))

        # 6) Primary & secondary changes
        prim = row.get("primary")
        sec = row.get("secondary")
        if pd.notna(prim):
            txt = f"Primary features include {prim}"
            if pd.notna(sec):
                txt += f", while secondary changes such as {sec} are present"
            parts.append(txt)

        # Combine all non-empty parts into one paragraph
        return ". ".join(p.rstrip(".") for p in parts) + "."
