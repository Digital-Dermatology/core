from pathlib import Path
from typing import Union

import pandas as pd
import torch
from PIL import Image

from ....src.datasets.base_dataset import BaseDataset


class Derm1MDataset(BaseDataset):
    """Derm1M dataset loader.

    This dataset loader handles the Derm1M dataset which contains dermatology images
    from various sources (edu, youtube, pubmed, twitter, IIYI, note, public, reddit).

    The dataset includes:
    - Main CSV files: Derm1M_v2_pretrain.csv and Derm1M_v2_validation.csv
    - Concept CSV: concept.csv (additional metadata with disease labels and skin concepts)
    - Image folders: edu/, youtube/, pubmed/, twitter/, IIYI/, note/, public/, reddit/
    """

    IMG_COL = "img_path"
    LBL_COL = "disease_label"

    def __init__(
        self,
        dataset_dir: Union[str, Path] = "/media/onix/data-repository/Derm1M",
        split: str = "all",  # "pretrain", "validation", or "all"
        transform=None,
        val_transform=None,
        return_path: bool = False,
        use_concept_csv: bool = True,
        **kwargs,
    ):
        """
        Initializes the Derm1M dataset.

        Parameters
        ----------
        dataset_dir : Union[str, Path]
            Directory containing the Derm1M dataset.
        split : str
            Which split to use: "pretrain", "validation", or "all" (combines both)
        transform : Union[callable, optional]
            Optional transform to be applied to the images during training.
        val_transform : Union[callable, optional]
            Optional transform to be applied to the images during validation.
        return_path : bool
            If the path of the image should be returned or not.
        use_concept_csv : bool
            Whether to merge concept.csv for additional metadata.
        """
        super().__init__(transform=transform, val_transform=val_transform, **kwargs)

        self.dataset_dir = self.check_path(dataset_dir)
        self.return_path = return_path
        self.split = split

        # Load the appropriate CSV file(s)
        if split == "pretrain":
            csv_path = self.dataset_dir / "Derm1M_v2_pretrain.csv"
            if not csv_path.exists():
                raise ValueError(f"CSV file not found: {csv_path}")
            self.meta_data = pd.read_csv(csv_path)
            self.meta_data["split"] = "pretrain"
        elif split == "validation":
            csv_path = self.dataset_dir / "Derm1M_v2_validation.csv"
            if not csv_path.exists():
                raise ValueError(f"CSV file not found: {csv_path}")
            self.meta_data = pd.read_csv(csv_path)
            self.meta_data["split"] = "validation"
        elif split == "all":
            # Load both pretrain and validation
            pretrain_path = self.dataset_dir / "Derm1M_v2_pretrain.csv"
            validation_path = self.dataset_dir / "Derm1M_v2_validation.csv"

            dfs = []
            if pretrain_path.exists():
                df_pretrain = pd.read_csv(pretrain_path)
                df_pretrain["split"] = "pretrain"
                dfs.append(df_pretrain)

            if validation_path.exists():
                df_validation = pd.read_csv(validation_path)
                df_validation["split"] = "validation"
                dfs.append(df_validation)

            if not dfs:
                raise ValueError(f"No CSV files found in {self.dataset_dir}")

            self.meta_data = pd.concat(dfs, ignore_index=True)
        else:
            raise ValueError(
                f"Invalid split: {split}. Choose 'pretrain', 'validation', or 'all'"
            )

        # Create full image paths
        self.meta_data[self.IMG_COL] = self.meta_data["filename"].apply(
            lambda x: str(self.dataset_dir / x)
        )

        # Add img_name column for consistency with other datasets
        self.meta_data["img_name"] = self.meta_data["filename"].apply(
            lambda x: Path(x).stem
        )

        # Merge with concept.csv if requested
        if use_concept_csv:
            concept_csv_path = self.dataset_dir / "concept.csv"
            if concept_csv_path.exists():
                df_concept = pd.read_csv(concept_csv_path)
                self.meta_data = self.meta_data.merge(
                    df_concept,
                    on="filename",
                    how="left",
                    suffixes=("", "_concept"),
                )
                # Use concept disease_label if available, otherwise use from main CSV
                if "disease_label_concept" in self.meta_data.columns:
                    self.meta_data[self.LBL_COL] = self.meta_data[self.LBL_COL].fillna(
                        self.meta_data["disease_label_concept"]
                    )

        # Fill missing disease labels with "no definitive diagnosis"
        self.meta_data[self.LBL_COL] = self.meta_data[self.LBL_COL].fillna(
            "no definitive diagnosis"
        )

        # Use caption as description
        if "caption" in self.meta_data.columns:
            self.meta_data["description"] = self.meta_data["caption"]
        else:
            self.meta_data["description"] = self.meta_data[self.LBL_COL]

        # Map disease_label to condition for consistency with other datasets
        self.meta_data["condition"] = self.meta_data[self.LBL_COL]

        # Clean up "No X information" values by replacing with NaN
        replacements = {
            "No body location information": None,
            "No age information": None,
            "No gender information": None,
            "No symptom information": None,
        }

        for col in ["body_location", "gender", "symptoms"]:
            if col in self.meta_data.columns:
                self.meta_data[col] = self.meta_data[col].replace(replacements)

        # Convert age to numeric, handling "No age information"
        if "age" in self.meta_data.columns:
            self.meta_data["age"] = self.meta_data["age"].replace(replacements)
            self.meta_data["age"] = pd.to_numeric(
                self.meta_data["age"], errors="coerce"
            )

        # Filter out rows where image file doesn't exist (optional, can be commented out)
        # self.meta_data = self.meta_data[
        #     self.meta_data[self.IMG_COL].apply(lambda x: Path(x).exists())
        # ]

        self.meta_data.reset_index(drop=True, inplace=True)

        # Create integer labels for classification tasks
        int_lbl, lbl_mapping = pd.factorize(self.meta_data[self.LBL_COL])
        self.meta_data[f"lbl_{self.LBL_COL}"] = int_lbl

        # Global configs
        self.classes = list(lbl_mapping)
        self.n_classes = len(self.classes)

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        img_path = self.meta_data.loc[self.meta_data.index[index], self.IMG_COL]

        try:
            image = Image.open(img_path)
            image = image.convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error loading image at {img_path}: {e}")

        if self.transform and self.training:
            image = self.transform(image)
        elif self.val_transform and not self.training:
            image = self.val_transform(image)

        diagnosis = self.meta_data.loc[
            self.meta_data.index[index], f"lbl_{self.LBL_COL}"
        ]

        if self.return_path:
            return image, img_path, int(diagnosis)
        else:
            return image, int(diagnosis)
