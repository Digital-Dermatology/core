import os
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import torch
from PIL import Image

from ....src.datasets.base_dataset import BaseDataset


class DermaConINLabel(Enum):
    MAIN_CLASS = "Main_class"
    SUB_CLASS = "Sub_class"
    DISEASE = "condition"


class DermaConINDataset(BaseDataset):
    """DermaCon-IN dataset - A dermatology dataset from India."""

    IMG_COL = "Image_name"
    LBL_COL = "Disease_label"

    def __init__(
        self,
        csv_file: Union[str, Path] = "/data/DermaCon-IN/METADATA/Skin_Metadata.csv",
        root_dir: Union[str, Path] = "/data/DermaCon-IN/DATASET",
        transform=None,
        val_transform=None,
        label_col: DermaConINLabel = DermaConINLabel.DISEASE,
        return_path: bool = False,
        data_quality_issues_list: Optional[Union[str, Path]] = None,
        return_embedding: bool = False,
        **kwargs,
    ):
        """
        Initializes the DermaCon-IN dataset.

        Parameters
        ----------
        csv_file : str
            Path to the csv file with metadata.
        root_dir : str
            Directory with all the images (DATASET folder containing DATASET_0 and DATASET_1).
        transform : Union[callable, optional]
            Optional transform to be applied to the images.
        label_col : DermaConINLabel
            Which label column to use (Main_class, Sub_class, or Disease_label).
        return_path : bool
            If the path of the image should be returned or not.
        """
        super().__init__(transform=transform, val_transform=val_transform, **kwargs)

        # Check if the dataset path exists
        self.csv_file = Path(csv_file)
        if not self.csv_file.exists():
            raise ValueError(f"CSV metadata path must exist, path: {self.csv_file}")
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise ValueError(f"Image path must exist, path: {self.root_dir}")

        # Load the metadata
        self.meta_data = pd.DataFrame(pd.read_csv(self.csv_file))

        # Determine which subfolder (DATASET_0 or DATASET_1) each image is in
        def find_image_path(image_name):
            """Find the full path to the image in either DATASET_0 or DATASET_1."""
            for subfolder in ["DATASET_0", "DATASET_1"]:
                full_path = os.path.join(self.root_dir, subfolder, image_name)
                if os.path.exists(full_path):
                    return os.path.join(subfolder, image_name)
            # Return default path if not found (will raise error on load)
            return os.path.join("DATASET_0", image_name)

        # Add full relative path column
        self.meta_data["img_path"] = self.meta_data[self.IMG_COL].apply(find_image_path)

        # Standardize column names for compatibility
        self.meta_data = self.meta_data.rename(columns={
            "Sex": "gender",
            "Age": "age",
            "Fitzpatrick": "fitzpatrick",
            "Body_part": "body_location",
            "Disease_label": "condition",
            "Subject_ID": "subject_id",
        })

        # Clean up fitzpatrick column (remove "FST " prefix)
        if "fitzpatrick" in self.meta_data.columns:
            self.meta_data["fitzpatrick"] = self.meta_data["fitzpatrick"].str.replace("FST ", "").str.strip()
            self.meta_data["fitzpatrick"] = pd.to_numeric(self.meta_data["fitzpatrick"], errors="coerce")

        # Clean up Monk skin tone column
        if "Monk_skin_tone" in self.meta_data.columns:
            self.meta_data["monk_skin_tone"] = self.meta_data["Monk_skin_tone"].str.replace("MST ", "").str.strip()
            self.meta_data["monk_skin_tone"] = pd.to_numeric(self.meta_data["monk_skin_tone"], errors="coerce")

        # Parse age ranges to median values
        def parse_age(age_str):
            """Convert age range like '10 - 20' to median value."""
            if pd.isna(age_str):
                return None
            try:
                # If already a number, return it
                return float(age_str)
            except (ValueError, TypeError):
                pass

            # Parse range like "10 - 20"
            age_str = str(age_str).strip()
            if " - " in age_str:
                parts = age_str.split(" - ")
                if len(parts) == 2:
                    try:
                        low = float(parts[0])
                        high = float(parts[1])
                        return (low + high) / 2
                    except ValueError:
                        pass
            return None

        self.meta_data["age"] = self.meta_data["age"].apply(parse_age)

        # Encode the label column
        label_col_name = label_col.value
        int_lbl, lbl_mapping = pd.factorize(self.meta_data[label_col_name])
        self.meta_data[label_col_name + "_name"] = self.meta_data[label_col_name]
        self.meta_data[label_col_name] = int_lbl

        # Add dataset description
        self.meta_data["dataset_desc"] = "DermaCon-IN"

        # Create descriptions
        def create_description(row):
            parts = []

            # Base description with condition
            if pd.notna(row.get("condition")):
                parts.append(f"This clinical image shows {row['condition']}")
            else:
                parts.append("This clinical image shows a skin condition")

            # Add body location
            if pd.notna(row.get("body_location")):
                parts.append(f"on the {row['body_location'].lower()}")

            # Add patient information
            if pd.notna(row.get("gender")):
                parts.append(f"for a {row['gender'].lower()} patient")

            # Add Fitzpatrick skin type
            if pd.notna(row.get("fitzpatrick")):
                try:
                    fitzpatrick_int = int(row["fitzpatrick"])
                    parts.append(f"with Fitzpatrick skin type {fitzpatrick_int}")
                except (ValueError, TypeError):
                    pass

            # Add age
            if pd.notna(row.get("age")):
                try:
                    age_val = int(row["age"])
                    parts.append(f"of age {age_val}")
                except (ValueError, TypeError):
                    pass

            description = " ".join(parts) + "."
            return " ".join(description.split())  # Clean up extra whitespace

        self.meta_data["description"] = self.meta_data.apply(create_description, axis=1)
        self.meta_data["description_short"] = self.meta_data["description"]

        # Remove data quality issues if file is given
        self.remove_data_quality_issues(data_quality_issues_list)
        self.meta_data.reset_index(drop=True, inplace=True)

        # Global configs
        self.LBL_COL = label_col.value
        self.return_path = return_path
        self.return_embedding = return_embedding
        self.classes = list(lbl_mapping)
        self.n_classes = len(self.classes)

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.meta_data.loc[self.meta_data.index[idx], "img_path"]
        img_name = os.path.join(self.root_dir, img_path)
        diagnosis = self.meta_data.loc[self.meta_data.index[idx], self.LBL_COL]

        if self.return_embedding:
            embedding = self.meta_data.loc[self.meta_data.index[idx], "embedding"]
            if self.return_path:
                return embedding, img_name, int(diagnosis)
            else:
                return embedding, int(diagnosis)

        image = Image.open(img_name)
        image = image.convert("RGB")
        if self.transform and self.training:
            image = self.transform(image)
        elif self.val_transform and not self.training:
            image = self.val_transform(image)

        if self.return_path:
            return image, img_name, int(diagnosis)
        else:
            return image, int(diagnosis)
