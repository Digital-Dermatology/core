import copy
import inspect
import json
import os
import random
from enum import Enum
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
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
        instances_file: Optional[str] = None,
    ):
        self.annotation_file = annotation_file
        self.image_dir = image_dir
        self.transform = transform
        self.mask_transform = make_mask_transform(transform)
        self.tokenizer = tokenizer
        self.loading_type = loading_type
        self.num_tasks = 4

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
                samples.append((image_id, image_path, caption))
        self.df = pd.DataFrame(samples, columns=["image_id", "image_path", "captions"])
        self.df.dropna(subset="captions", inplace=True)

        # COCO annotations (i.e., bounding boxes)
        self.coco_captions = COCO(annotation_file)
        self.coco_instances = None
        if instances_file:
            self.coco_instances = COCO(instances_file)
            # select only the valid samples, because the others have no annotations
            valid_samples = list(self.coco_instances.imgToAnns.keys())
            self.df = self.df[self.df["image_id"].isin(valid_samples)]

        # this optimizes the loading afterwards
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
                    list(self.df["captions"].values),
                    padding="longest",
                    return_tensors="pt",
                )
            else:
                self.tokens = self.tokenizer(list(self.df["captions"].values))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        _, image_path, caption = self.df.iloc[idx]

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

    def get_batch_image_masks_and_instructions(self, indices: list):
        l_masks = []
        for idx in indices:
            l_masks.append(self.get_single_task(idx=idx))

        task_ids = torch.Tensor([x[0] for x in l_masks]).int()
        instructions = [x[1] for x in l_masks]
        masks = torch.stack([x[2] for x in l_masks]).unsqueeze(1).float()
        captions = [x[3] for x in l_masks]

        # tokenize the captions
        if self.tokenizer is not None:
            captions = self.tokenizer(captions)

        return task_ids, instructions, masks, captions

    def get_all_tasks(self, idx: int, print_captions: bool = False):
        all_tasks = []
        for task_id in range(self.num_tasks):
            all_tasks.append(
                self.get_single_task(
                    idx=idx,
                    print_captions=print_captions,
                    task_id=task_id,
                )
            )
        return all_tasks

    def get_single_task(
        self,
        idx: int,
        print_captions: bool = False,
        task_id: Optional[int] = None,
    ):
        """
        Generates a single random task and its corresponding mask and caption for a given index.
        This is designed to be called from __getitem__.
        """
        if self.coco_captions is None or self.coco_instances is None:
            raise ValueError("Caption and instance metadata not available.")

        # get image and instance info
        img_id = self.df.iloc[idx].image_id
        img_info = self.coco_captions.loadImgs([img_id])[0]
        height, width = img_info["height"], img_info["width"]
        ann_ids = self.coco_instances.getAnnIds(imgIds=[img_id], iscrowd=None)
        annotations = self.coco_instances.loadAnns(ann_ids)
        if not annotations:
            # return a dummy task if no annotations are found
            dummy_mask = np.ones((height, width), dtype=np.uint8)
            return 0, "no objects", self.mask_transform(dummy_mask), "no objects"

        if print_captions:
            cap_ids = self.coco_captions.getAnnIds(imgIds=[img_id])
            caps = self.coco_captions.loadAnns(cap_ids)
            captions = [cap["caption"] for cap in caps]
            print(f"Image {img_id} captions:")
            for idx, text in enumerate(captions, 1):
                print(f"  {idx}. {text}")

        # select random task if none is provided
        if task_id is None:
            task_id = random.randint(0, self.num_tasks - 1)

        # generate only the required caption and mask
        if task_id == 0:
            # caption
            central_annotation = get_central_instance(annotations, width, height)
            central_category = central_annotation["category_id"]
            central_category_name = self.coco_instances.cats[central_category]["name"]
            caption_central = f"this image shows a {central_category_name}"
            # "all 1's" mask
            mask = np.ones((height, width), dtype=np.uint8)
            return (
                task_id,
                "focus on the central object",
                self.mask_transform(mask),
                caption_central,
            )
        elif task_id == 1:
            # caption
            central_annotation = get_central_instance(annotations, width, height)
            central_category = central_annotation["category_id"]
            central_category_name = self.coco_instances.cats[central_category]["name"]
            caption_central = f"this image shows a {central_category_name}"
            # mask showing only the central object
            central_annotations = [
                ann for ann in annotations if ann["category_id"] == central_category
            ]
            mask = union_masks(self.coco_instances, central_annotations, height, width)
            return (
                task_id,
                "focus on the entire region of the central object",
                self.mask_transform(mask),
                caption_central,
            )
        elif task_id == 2:
            # caption
            instance_cat_names = list(
                set(
                    [
                        self.coco_instances.cats[x["category_id"]]["name"]
                        for x in annotations
                    ]
                )
            )
            caption_objects = f'this image shows {", ".join(instance_cat_names)}'
            # mask showing all objects in the image
            mask_union = union_masks(self.coco_instances, annotations, height, width)
            return (
                task_id,
                "focus on the entire region of all objects",
                self.mask_transform(mask_union),
                caption_objects,
            )
        elif task_id == 3:
            # caption
            instance_cat_names = list(
                set(
                    [
                        self.coco_instances.cats[x["category_id"]]["name"]
                        for x in annotations
                    ]
                )
            )
            caption_objects = f'this image shows {", ".join(instance_cat_names)}'
            # mask showing all objects in the image
            mask_union = union_masks(self.coco_instances, annotations, height, width)
            return (
                task_id,
                "Separate foreground objects from background",
                self.mask_transform(mask_union),
                caption_objects,
            )

        else:
            raise ValueError(f"Unknown task ID: {task_id}")


def get_central_instance(anns, img_width, img_height):
    """Select the annotation whose bounding-box centroid is closest to the image center."""
    # TODO: I think this should not be the "center" but just the biggest?
    cx, cy = img_width / 2, img_height / 2
    best, best_dist = None, float("inf")
    for ann in anns:
        x, y, w, h = ann["bbox"]
        cent_x, cent_y = x + w / 2, y + h / 2
        dist = (cent_x - cx) ** 2 + (cent_y - cy) ** 2
        if dist < best_dist:
            best_dist, best = dist, ann
    return best


def union_masks(coco, anns, height, width):
    """Union all instance masks in anns into a single binary mask."""
    mask_union = np.zeros((height, width), dtype=np.uint8)
    for ann in anns:
        mask_union = np.logical_or(mask_union, coco.annToMask(ann)).astype(np.uint8)
    return mask_union


def make_mask_transform(image_transform: transforms.Compose) -> transforms.Compose:
    mask_transforms = [transforms.Lambda(lambda pic: Image.fromarray(pic))]
    for t in image_transform.transforms:
        # deep‐copy so we don’t clobber the original
        t_mask = copy.deepcopy(t)

        # if it’s a resize, force nearest‐neighbor
        if isinstance(t_mask, transforms.Resize):
            t_mask.interpolation = InterpolationMode.NEAREST

        # if it’s a CenterCrop (or any crop) it has no interpolation
        # but we still want the same size arg
        elif isinstance(t_mask, transforms.CenterCrop):
            # nothing to change here
            pass

        # ToTensor on masks will normalize to [0,1] float – you probably
        # want integer labels instead, so convert differently:
        elif isinstance(t_mask, transforms.ToTensor):
            # replace with a lambda that just turns your PIL mask into LongTensor
            t_mask = transforms.Lambda(
                lambda pic: torch.from_numpy(np.array(pic, dtype=np.int64))
            )

        # drop any color/brightness etc transforms:
        else:
            continue

        mask_transforms.append(t_mask)

    return transforms.Compose(mask_transforms)
