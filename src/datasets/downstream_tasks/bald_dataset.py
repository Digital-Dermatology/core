from pathlib import Path
from typing import Sequence, Union

import pandas as pd

from ....src.datasets.generic_image_dataset import GenericImageDataset
from ._isic_common import normalize_isic_metadata


class BALDDataset(GenericImageDataset):
    """BALD (BRAAFF-Annotated Acral Lesions Dataset) image dataset.

    666 dermatoscopic images of acral lesions (144 melanomas, 522 nevi) from
    seven academic centers in Austria, Greece, France, Italy and Japan; hosted
    on the ISIC archive. Expects the ISIC-archive metadata export (same schema
    as MSKCC/HIBA) with images under ``<dataset_dir>/images``.
    """

    IMG_COL = "img_path"
    LBL_COL = "diagnosis"

    def __init__(
        self,
        dataset_dir: Union[str, Path] = "data/dataset/",
        bald_meta_name: Union[str, Path] = "bald.csv",
        transform=None,
        val_transform=None,
        return_path: bool = False,
        image_extensions: Sequence = ("*.jpg", "*.JPG", "*.png", "*.JPEG"),
        **kwargs,
    ):
        """
        Initializes the BALD dataset.

        Parameters
        ----------
        dataset_dir : str
            Directory with all the images.
        bald_meta_name : str
            Name of the metadata CSV file.
        transform : Union[callable, optional]
            Optional transform to be applied to the images.
        val_transform : Union[callable, optional]
            Optional transform to be applied to the images when in validation mode.
        return_path : bool
            If the path of the image should be returned or not.
        image_extensions : Sequence
            Sequence of image file extensions to search for.
        """
        super().__init__(
            dataset_dir=dataset_dir,
            transform=transform,
            val_transform=val_transform,
            return_path=return_path,
            image_extensions=image_extensions,
            **kwargs,
        )

        # Load metadata CSV directly and normalize the raw ISIC-archive schema
        # (isic_id, diagnosis_1..5, anatom_site_*) to the flat loader schema.
        self.meta_data = pd.read_csv(self.dataset_dir / bald_meta_name)
        self.meta_data = normalize_isic_metadata(self.meta_data)

        # Create img_path from image column
        self.meta_data["img_path"] = self.meta_data["image"].apply(
            lambda x: str(self.dataset_dir / "images" / x)
        )

        self.meta_data = self.meta_data[self.meta_data["img_path"].notna()]
        self.meta_data.reset_index(drop=True, inplace=True)

        int_lbl, lbl_mapping = pd.factorize(self.meta_data["diagnosis"])
        self.meta_data["lbl_diagnosis"] = int_lbl

        def create_description(row):
            parts = []

            # Basic info: sex and age (both optional in the ISIC export)
            sex = row.get("sex")
            age = row.get("age_approx")
            sex_text = str(sex) if pd.notna(sex) else None
            age_text = (
                f"approximately {int(age)} years old" if pd.notna(age) else None
            )
            if sex_text and age_text:
                parts.append(f"A {sex_text} patient, {age_text},")
            elif sex_text:
                parts.append(f"A {sex_text} patient")
            elif age_text:
                parts.append(f"A patient, {age_text},")
            else:
                parts.append("A patient")

            # Anatomical site (acral: soles / palms / nails)
            site = (
                row.get("anatom_site_general")
                if pd.notna(row.get("anatom_site_general"))
                else "acral site"
            )
            parts.append(f"presents with an acral lesion on the {site}.")

            # Confirmation type
            confirm = row.get("diagnosis_confirm_type")
            if pd.notna(confirm):
                parts.append(f"Confirmed via {confirm}.")

            # Diagnosis
            diagnosis = row.get("diagnosis")
            if pd.notna(diagnosis):
                parts.append(f"Diagnosis: {diagnosis}.")

            # Image type
            image_type = (
                row.get("image_type")
                if pd.notna(row.get("image_type"))
                else "dermoscopic"
            )
            parts.append(f"Image type: {image_type}.")

            return " ".join(parts)

        self.meta_data["description"] = self.meta_data.apply(create_description, axis=1)
        self.meta_data = self.meta_data.rename(
            columns={
                "diagnosis": "condition",
                "anatom_site_general": "body_location",
                "sex": "gender",
                "age_approx": "age",
                "fitzpatrick_skin_type": "fitzpatrick",
            },
        )
        self.meta_data["dataset_desc"] = "BALD"

        # Global configs
        self.classes = list(lbl_mapping)
        self.n_classes = len(self.classes)
