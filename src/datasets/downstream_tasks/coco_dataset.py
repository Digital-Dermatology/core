import json
import os
from enum import Enum

import torch
from torch.utils.data import Dataset
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
        self.index_samples()
        # create TurboJPEG object for image reading
        self.jpeg_reader = TurboJPEG()

    def index_samples(self) -> None:
        with open(self.annotation_file, "r") as f:
            data = json.load(f)

        self.image_dict = {img["id"]: img["file_name"] for img in data["images"]}
        self.samples = []
        for ann in data["annotations"]:
            image_id = ann["image_id"]
            # Make sure the image exists in our mapping.
            if image_id in self.image_dict:
                file_name = self.image_dict[image_id]
                image_path = os.path.join(self.image_dir, file_name)
                caption = ann["caption"]
                self.samples.append((image_path, caption))
        if self.tokenizer:
            self.tokens = self.tokenizer(
                [x[1] for x in self.samples], padding="longest", return_tensors="pt"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, caption = self.samples[idx]

        if (
            self.loading_type == LoadingType.STANDARD
            or self.loading_type == LoadingType.IMG_ONLY
        ):
            # image = Image.open(image_path).convert('RGB')
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
            image = self.jpeg_reader.decode(file.read())
        if len(image.shape) == 2:
            image = image[...,]
        return torch.Tensor(image).permute(2, 0, 1)

    @staticmethod
    def collate_fn(batch):
        images, captions = zip(*batch)
        images = torch.stack(images, dim=0)
        if isinstance(captions[0], dict):
            # Batch tokenized captions: each caption is a dict (e.g., input_ids, attention_mask).
            # We assume that all captions have the same keys.
            batch_captions = {
                key: torch.stack([caption[key] for caption in captions], dim=0)
                for key in captions[0]
            }
        else:
            batch_captions = torch.stack(captions, dim=0)
        return images, batch_captions
