import inspect
import os
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from turbojpeg import TurboJPEG

from src.core.src.datasets.downstream_tasks.coco_dataset import LoadingType


class Flickr30kDataset(Dataset):

    def __init__(
        self,
        root_dir,
        meta_path,
        split: Optional[Union[str, List[str]]] = "train",
        transform=None,
        tokenizer=None,
        loading_type: LoadingType = LoadingType.STANDARD,
    ):
        self.root_dir = root_dir
        self.meta_path = meta_path
        self.split = split
        self.transform = transform
        self.tokenizer = tokenizer
        self.loading_type = loading_type

        self.df = pd.read_csv(self.meta_path, delimiter="|")
        self.df.columns = [x.strip() for x in self.df.columns]

        split_annotations = []
        for split in ["train", "val", "test"]:
            with open(os.path.join(self.root_dir, f"{split}.txt")) as file:
                split_indices = [line.rstrip() for line in file]
            split_indices = [(x, split) for x in split_indices]
            split_annotations += split_indices
        df_split = pd.DataFrame(split_annotations, columns=["image_name", "split"])
        df_split["image_name"] = df_split["image_name"] + ".jpg"
        df_split["image_path"] = df_split["image_name"].apply(
            lambda x: os.path.join(self.root_dir, "flickr30k_images", x)
        )
        self.df = self.df.merge(df_split, on="image_name", how="left")
        self.df.dropna(subset="comment", inplace=True)
        # select the correct dataset
        if self.split is not None:
            if type(self.split) is str:
                self.df = self.df[self.df["split"] == self.split]
            else:
                self.df = self.df[self.df["split"].isin(self.split)]
            self.df.reset_index(drop=True, inplace=True)
        self.apply_tokenizer()

        try:
            # create TurboJPEG object for (fast) image reading
            self.jpeg_reader = TurboJPEG()
        except RuntimeError as e:
            self.jpeg_reader = None
            print(f"Failed to create TurboJPEG object falling back on PIL: {e}")

    def apply_tokenizer(self) -> None:
        if self.tokenizer:
            arguments = inspect.getfullargspec(self.tokenizer).args
            if "padding" in arguments and "return_tensors" in arguments:
                self.tokens = self.tokenizer(
                    list(self.df["comment"].values),
                    padding="longest",
                    return_tensors="pt",
                )
            else:
                self.tokens = self.tokenizer(list(self.df["comment"].values))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        caption = self.df.iloc[idx].comment.strip()
        image_path = self.df.iloc[idx].image_path

        if (
            self.loading_type == LoadingType.STANDARD
            or self.loading_type == LoadingType.IMG_ONLY
        ):
            image = self.load_image_turbo_jpeg(image_path)
            if self.transform:
                image = self.transform(image)
        else:
            image = torch.Tensor(0)

        if (
            self.loading_type == LoadingType.STANDARD
            or self.loading_type == LoadingType.TXT_ONLY
        ):
            if self.tokenizer:
                if type(self.tokens) is torch.Tensor:
                    caption = self.tokens[idx]
                else:
                    caption = {k: v[idx] for (k, v) in self.tokens.items()}
        else:
            caption = torch.Tensor(0)

        return image, caption

    def load_image_turbo_jpeg(self, f):
        if self.jpeg_reader is None:
            image = self.load_image_PIL(f=f)
        else:
            with open(f, "rb") as file:
                try:
                    image = self.jpeg_reader.decode(file.read())
                    image = Image.fromarray(image)
                except OSError:
                    # fall back to PIL loading when there is a problem
                    # likely not a JPEG image
                    print(
                        f"Failed to read file with TurboJPEG falling back on PIL: {f}"
                    )
                    image = self.load_image_PIL(f=f)
        image = image.convert("RGB")
        return image

    def load_image_PIL(self, f):
        image = Image.open(f)
        return image
