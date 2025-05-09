import json
import os
from enum import Enum

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from turbojpeg import TurboJPEG


class LoadingType(Enum):
    STANDARD = 0
    IMG_ONLY = 1
    TXT_ONLY = 2


class CocoCaptionDataset(Dataset):
    def __init__(
        self,
        annotation_file: str,
        image_dir: str,
        transform=None,
        tokenizer=None,
        loading_type: LoadingType = LoadingType.STANDARD,
    ):
        self.annotation_file = annotation_file
        self.image_dir = image_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.loading_type = loading_type

        with open(self.annotation_file, "r") as f:
            data = json.load(f)

        self.image_dict = {img["id"]: img["file_name"] for img in data["images"]}
        samples = []
        for ann in data["annotations"]:
            image_id = ann["image_id"]
            # Make sure the image exists in our mapping.
            if image_id in self.image_dict:
                file_name = self.image_dict[image_id]
                image_path = os.path.join(self.image_dir, file_name)
                caption = ann["caption"]
                samples.append((image_path, caption))
        self.df = pd.DataFrame(samples, columns=["image_path", "captions"])
        self.df.dropna(subset="captions", inplace=True)
        self.apply_tokenizer()

        # create TurboJPEG object for image reading
        self.jpeg_reader = TurboJPEG()

    def apply_tokenizer(self) -> None:
        if self.tokenizer:
            self.tokens = self.tokenizer(
                list(self.df["captions"].values),
                padding="longest",
                return_tensors="pt",
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        image_path, caption = self.df.iloc[idx]

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
                caption = {k: v[idx] for (k, v) in self.tokens.items()}
        else:
            caption = torch.Tensor(0)

        return image, caption

    def load_image_turbo_jpeg(self, f):
        with open(f, "rb") as file:
            try:
                image = self.jpeg_reader.decode(file.read())
            except OSError:
                # fall back to PIL loading when there is a problem
                # likely not a JPEG image
                print(f"Failed to read file with TurboJPEG falling back on PIL: {f}")
                image = Image.open(f)
                image = image.convert("RGB")
                image = np.array(image)
        if len(image.shape) == 2:
            image = image[...,]
        return transforms.ToTensor()(image)
