from pathlib import Path
from typing import Sequence, Union

import pandas as pd

from ....src.datasets.generic_image_dataset import GenericImageDataset


class ISICDataset(GenericImageDataset):
    """ISIC image dataset."""

    IMG_COL = "img_path"
    LBL_COL = "diagnosis"

    def __init__(
        self,
        dataset_dir: Union[str, Path] = "data/dataset/",
        isic_train_meta_name: Union[str, Path] = "metadata.csv",
        transform=None,
        val_transform=None,
        return_path: bool = False,
        image_extensions: Sequence = ("*.png", "*.jpg", "*.JPEG"),
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
        df_meta = pd.read_csv(self.dataset_dir / isic_train_meta_name)
        self.meta_data = self.meta_data.merge(
            df_meta,
            left_on="img_name",
            right_on="isic_id",
            how="outer",
        )
        self.meta_data = self.meta_data[self.meta_data["img_path"].notna()]
        self.meta_data.reset_index(drop=True, inplace=True)

        int_lbl, lbl_mapping = pd.factorize(self.meta_data["diagnosis_1"])
        self.meta_data["lbl_diagnosis"] = int_lbl

        def create_description(row):
            parts = []

            # Basic info: sex and age
            sex = row.get("sex") if pd.notna(row.get("sex")) else "patient"
            age = row.get("age_approx")
            age_text = f"{int(age)} years old" if pd.notna(age) else "age unknown"
            parts.append(f"A {sex} patient, approximately {age_text},")

            # Anatomical site
            site = (
                row.get("anatom_site_general")
                if pd.notna(row.get("anatom_site_general"))
                else "unknown site"
            )
            parts.append(f"presents with a lesion on the {site}.")

            # Confirmation type, only if provided
            confirm = row.get("diagnosis_confirm_type")
            if pd.notna(confirm):
                parts.append(f"Confirmed via {confirm}.")

            # Concomitant biopsy note
            if row.get("concomitant_biopsy"):
                parts.append("Concomitant biopsy performed.")

            # Diagnoses, only if any are present
            diagnoses = [
                d
                for d in [
                    row.get("diagnosis_1"),
                    row.get("diagnosis_2"),
                    row.get("diagnosis_3"),
                    row.get("diagnosis_4"),
                    row.get("diagnosis_5"),
                ]
                if pd.notna(d)
            ]
            if diagnoses:
                parts.append(f"Diagnosis details: {', '.join(diagnoses)}.")

            # Image type
            image_type = (
                row.get("image_type") if pd.notna(row.get("image_type")) else "N/A"
            )
            parts.append(f"Image type: {image_type}.")

            # Combine parts with spaces
            return " ".join(parts)

        self.meta_data["description"] = self.meta_data.apply(create_description, axis=1)
        self.meta_data = self.meta_data.rename(
            columns={
                "diagnosis_1": "condition",
                "anatom_site_general": "body_location",
                "sex": "gender",
                "age_approx": "age",
            },
        )

        # Global configs
        self.classes = list(lbl_mapping)
        self.n_classes = len(self.classes)
