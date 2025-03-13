import json
from pathlib import Path
from typing import Sequence, Union

import numpy as np
import pandas as pd

from ....src.datasets.generic_image_dataset import GenericImageDataset


class ImageNet1kDataset(GenericImageDataset):
    """ImageNet-1k image dataset."""

    def __init__(
        self,
        dataset_dir: Union[str, Path] = "data/dataset/",
        cleanlab_annot: Union[str, Path] = "cleanlab_imagenet_label_errors.csv",
        val_solutions: Union[str, Path] = "ImageNet-1k/ILSVRC/LOC_val_solution.csv",
        mapping_path: Union[str, Path] = "ImageNet-1k/ILSVRC/LOC_synset_mapping.txt",
        reassessed_lbl_path: Union[None, str, Path] = None,
        clustered_lbl_path: Union[None, str, Path] = None,
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
        # Merge with ImageNet Annotations
        val_solutions = self.check_path(val_solutions)
        mapping_path = self.check_path(mapping_path)
        df_m = pd.read_csv(val_solutions)
        df_m["class"] = df_m["PredictionString"].apply(lambda x: x.split(" ")[0])
        # Creation of mapping dictionaries to obtain the image classes
        class_mapping_dict = {}
        class_mapping_dict_number = {}
        mapping_class_to_number = {}
        mapping_number_to_class = {}
        i = 0
        with open(mapping_path, "r") as mapping_file:
            for line in mapping_file:
                class_mapping_dict[line[:9].strip()] = line[9:].strip()
                class_mapping_dict_number[i] = line[9:].strip()
                mapping_class_to_number[line[:9].strip()] = i
                mapping_number_to_class[i] = line[:9].strip()
                i += 1
        df_m["class_name"] = df_m["class"].apply(class_mapping_dict.get)
        df_m["class_nr"] = df_m["class"].apply(mapping_class_to_number.get)
        self.meta_data = self.meta_data.merge(
            right=df_m,
            left_on="img_name",
            right_on="ImageId",
        )

        # Merge with CleanLab Annotations
        if cleanlab_annot is not None:
            cleanlab_annot = self.check_path(cleanlab_annot)
            df_m = pd.read_csv(cleanlab_annot, index_col=0)
            df_m["img_name"] = df_m["url"].apply(lambda x: Path(x).stem)
            df_m["origin"] = "CleanLab"
            self.meta_data = self.meta_data.merge(df_m, on="img_name", how="left")
            self.meta_data["lbl_err_cat"] = self.meta_data["mturk"].apply(
                lambda x: ImageNet1kDataset.categorize_error(x)
            )

        # Merge with Reassessed labels (Beyer et. al)
        self.meta_data = self.meta_data.sort_values(by="img_name")
        if reassessed_lbl_path is not None:
            reassessed_lbl_path = self.check_path(reassessed_lbl_path)
            with open(reassessed_lbl_path) as f:
                reassessed_labels = json.load(f)
            self.meta_data["reassessed_labels"] = reassessed_labels

        # Combined labels to 488 classes
        if clustered_lbl_path is not None:
            clustered_lbl_path = self.check_path(clustered_lbl_path)
            with open("../assets/clustered_imagenet_labels.json") as f:
                clustered_labels = json.load(f)
            self.meta_data.rename(columns={"lbl_diagnosis": "class_nr"}, inplace=True)
            self.meta_data["clustered_lbl"] = self.meta_data["class"].apply(
                clustered_labels.get
            )
            int_lbl, lbl_mapping = pd.factorize(self.meta_data["clustered_lbl"])
            self.meta_data["lbl_diagnosis"] = int_lbl
            self.classes = list(lbl_mapping)
        else:
            self.classes = list(class_mapping_dict_number.values())

        # Global configs
        self.LBL_COL = "class_nr"
        self.n_classes = len(self.classes)

    @staticmethod
    def categorize_error(data: str):
        """
        Confident learning categorizes into:
        (1) “correctable”, where a majority agree on the CL-predicted label;
        (2) “multi-label”, where a majority agree on both labels appearing;
        (3) “neither”, where a majority agree on neither label appearing;
        (4) “non-agreement”, a catch-all category for when there is no majority.
        """
        if str(data) == "nan":
            return np.nan
        data = eval(data)

        # make sure there is a majority
        total = sum(data.values())
        majority_threshold = total / 2
        majority_keys = [
            key for key, value in data.items() if value > majority_threshold
        ]

        if len(majority_keys) == 0:
            return "non-agreement"
        elif len(majority_keys) == 2:
            return "multi-label"
        else:
            # find the key with the majority value if it exists
            majority_key = None
            for key, value in data.items():
                if value > majority_threshold:
                    majority_key = key
                    break

            rename_dict = {
                "given": "non error",
                "guessed": "correctable",
                "neither": "neither",
                "both": "multi-label",
            }
            return rename_dict.get(majority_key)
