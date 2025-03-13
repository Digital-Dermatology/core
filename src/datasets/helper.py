from enum import Enum
from pathlib import Path

import numpy as np
from loguru import logger
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.datasets import CIFAR10

from ...src.datasets.base_dataset import BaseDataset
from ...src.datasets.downstream_tasks.celeba_dataset import CelebADataset
from ...src.datasets.downstream_tasks.chest_xray_covid_dataset import (
    ChestXrayCovidDataset,
)
from ...src.datasets.downstream_tasks.chexpert_dataset import CheXpertDataset
from ...src.datasets.downstream_tasks.ddi_dataset import DDIDataset
from ...src.datasets.downstream_tasks.derm7pt_dataset import Derm7ptDataset
from ...src.datasets.downstream_tasks.fitzpatrick17_dataset import Fitzpatrick17kDataset
from ...src.datasets.downstream_tasks.food101_dataset import Food101Dataset
from ...src.datasets.downstream_tasks.ham10000_dataset import HAM10000Dataset
from ...src.datasets.downstream_tasks.imagenet_1k_dataset import ImageNet1kDataset
from ...src.datasets.downstream_tasks.isic_2019_dataset import ISIC2019Dataset
from ...src.datasets.downstream_tasks.isic_2024_dataset import ISIC2024Dataset
from ...src.datasets.downstream_tasks.med_node_dataset import MedNodeDataset
from ...src.datasets.downstream_tasks.oxford_flowers102_dataset import (
    OxfordFlower102Dataset,
)
from ...src.datasets.downstream_tasks.pad_ufes_20_dataset import PADUFES20Dataset
from ...src.datasets.downstream_tasks.passion_dataset import PASSIONDataset
from ...src.datasets.downstream_tasks.pcam_dataset import PatchCamelyonDataset
from ...src.datasets.downstream_tasks.ph2_dataset import PH2Dataset
from ...src.datasets.downstream_tasks.sd_128_dataset import SD128Dataset
from ...src.datasets.downstream_tasks.stl_dataset import STL10Dataset
from ...src.datasets.downstream_tasks.vindr_bodypart_xr import VinDrBodyPartXRDataset
from ...src.datasets.encrypted_image_dataset import EncryptedImageDataset
from ...src.datasets.generic_image_dataset import GenericImageDataset


class DatasetName(Enum):
    MED_NODE = "MED-NODE"
    PH2 = "PH2"
    DDI = "DDI"
    DERM7PT = "Derm7pt"
    PAD_UFES_20 = "pad_ufes_20"
    SD_128 = "SD-128"
    HAM10000 = "ham10000"
    FITZPATRICK17K = "fitzpatrick17k"
    ISIC_2019 = "ISIC_2019"
    ISIC_2024 = "ISIC_2024"
    PASSION = "passion"

    CHEXPERT = "CheXpert"
    CHEST_XRAY_COVID = "Chest_Xray_COVID"
    VINDR_BODY_PART_XR = "VinDr_BodyPartXR"
    CELEB_A = "CelebA"
    PCAM = "PatchCamelyon"

    IMAGENET_1K = "ImageNet-1k"
    FOOD_101 = "FOOD_101"
    FLOWER_102 = "OxfordFlower102"
    STL_10 = "STL_10"
    CIFAR10 = "CIFAR10"
    CIFAR10H = "CIFAR10H"

    GENERIC = "generic"
    ENCRYPTED_GENERIC = "encrypted_generic"


def get_dataset(
    dataset_name: DatasetName,
    dataset_path: Path = Path("../data/"),
    batch_size: int = 128,
    transform=None,
    return_loader: bool = True,
    **kwargs,
) -> BaseDataset:
    if dataset_name == DatasetName.CHEST_XRAY_COVID:
        dataset_path = dataset_path / "Chest-Xray8-COVID-19/"
        dataset = ChestXrayCovidDataset(
            dataset_path,
            transform=transform,
            return_path=True,
            **kwargs,
        )

    elif dataset_name == DatasetName.DDI:
        dataset_path = dataset_path / "DDI/"
        csv_path = dataset_path / "ddi_metadata.csv"
        root_path = dataset_path / "images"
        dataset = DDIDataset(
            csv_path,
            root_path,
            transform=transform,
            return_path=True,
            **kwargs,
        )
    elif dataset_name == DatasetName.FITZPATRICK17K:
        dataset_path = dataset_path / "fitzpatrick17k/"
        csv_path = dataset_path / "fitzpatrick17kv2.csv"
        dataset = Fitzpatrick17kDataset(
            csv_path,
            dataset_path,
            transform=transform,
            return_path=True,
            **kwargs,
        )
    elif dataset_name == DatasetName.PAD_UFES_20:
        dataset_path = dataset_path / "PAD-UFES-20/"
        csv_path = dataset_path / "metadata.csv"
        root_path = dataset_path / "images"
        dataset = PADUFES20Dataset(
            csv_path,
            root_path,
            transform=transform,
            return_path=True,
            **kwargs,
        )
    elif dataset_name == DatasetName.HAM10000:
        dataset_path = dataset_path / "HAM10000/"
        csv_path = dataset_path / "HAM10000_metadata.csv"
        test_csv_path = dataset_path / "ISIC2018_Task3_Test_GroundTruth.csv"
        dataset = HAM10000Dataset(
            csv_path,
            dataset_path,
            transform=transform,
            return_path=True,
            test_csv_file=test_csv_path,
            **kwargs,
        )
    elif dataset_name == DatasetName.PASSION:
        dataset = PASSIONDataset(
            dataset_dir=dataset_path,
            meta_data_file="PASSION_traininready.csv",
            transform=transform,
            return_path=True,
            **kwargs,
        )
    elif dataset_name == DatasetName.ISIC_2019:
        dataset_path = dataset_path / "ISIC_2019/"
        dataset = ISIC2019Dataset(
            dataset_path,
            transform=transform,
            return_path=True,
            **kwargs,
        )
    elif dataset_name == DatasetName.ISIC_2024:
        dataset_path = dataset_path / "ISIC_2024/"
        dataset = ISIC2024Dataset(
            dataset_path,
            transform=transform,
            return_path=True,
            **kwargs,
        )
    elif dataset_name == DatasetName.FOOD_101:
        dataset = Food101Dataset(
            dataset_path,
            transform=transform,
            return_path=True,
            **kwargs,
        )
    elif dataset_name == DatasetName.IMAGENET_1K:
        dataset = ImageNet1kDataset(
            dataset_path,
            transform=transform,
            return_path=True,
            **kwargs,
        )
    elif dataset_name == DatasetName.CELEB_A:
        dataset = CelebADataset(
            dataset_path,
            transform=transform,
            return_path=True,
            **kwargs,
        )
    elif dataset_name == DatasetName.CHEXPERT:
        dataset = CheXpertDataset(
            dataset_path,
            transform=transform,
            return_path=True,
            **kwargs,
        )
    elif dataset_name == DatasetName.VINDR_BODY_PART_XR:
        dataset = VinDrBodyPartXRDataset(
            dataset_path,
            transform=transform,
            return_path=True,
            **kwargs,
        )
    elif dataset_name == DatasetName.PCAM:
        dataset = PatchCamelyonDataset(
            dataset_path,
            transform=transform,
            return_path=True,
            **kwargs,
        )
    elif dataset_name == DatasetName.GENERIC:
        dataset = GenericImageDataset(
            dataset_path,
            transform=transform,
            return_path=True,
            **kwargs,
        )
    elif dataset_name == DatasetName.ENCRYPTED_GENERIC:
        dataset = EncryptedImageDataset(
            dataset_path,
            transform=transform,
            return_path=True,
            **kwargs,
        )
    elif dataset_name == DatasetName.PH2:
        dataset = PH2Dataset(
            dataset_path / "PH2Dataset/PH2 Dataset images/",
            csv_file=dataset_path / "PH2Dataset/PH2_dataset.xlsx",
            transform=transform,
            return_path=True,
            **kwargs,
        )
    elif dataset_name == DatasetName.DERM7PT:
        dataset = Derm7ptDataset(
            dataset_path / "derm7pt/images/",
            csv_file=dataset_path / "derm7pt/meta/meta.csv",
            transform=transform,
            return_path=True,
            **kwargs,
        )
    elif dataset_name == DatasetName.MED_NODE:
        dataset = MedNodeDataset(
            dataset_path / "MED-NODE/",
            transform=transform,
            return_path=True,
            **kwargs,
        )
    elif dataset_name == DatasetName.SD_128:
        dataset = SD128Dataset(
            dataset_path / "SD-128/images/",
            transform=transform,
            return_path=True,
            **kwargs,
        )
    elif dataset_name == DatasetName.FLOWER_102:
        dataset = OxfordFlower102Dataset(
            dataset_path / "Oxford_Flowers102/",
            transform=transform,
            return_path=True,
            **kwargs,
        )
    elif dataset_name == DatasetName.STL_10:
        dataset = STL10Dataset(
            str(dataset_path / "STL"),
            transform=transform,
            download=True,
            split="train",
            **kwargs,
        )
    elif dataset_name == DatasetName.CIFAR10:
        train = CIFAR10(
            str(dataset_path / "CIFAR10"),
            transform=transform,
            download=True,
            train=True,
        )
        test = CIFAR10(
            str(dataset_path / "CIFAR10"),
            transform=transform,
            download=True,
            train=False,
        )
        dataset = ConcatDataset([train, test])
    elif dataset_name == DatasetName.CIFAR10H:
        dataset = CIFAR10(
            str(dataset_path / "CIFAR10"),
            transform=transform,
            download=True,
            train=False,
        )
        dataset.human_annotations = np.load(
            str(dataset_path / "CIFAR10" / "cifar10h-counts.npy")
        )
    else:
        raise ValueError(f"Unknown dataset: {str(dataset_name)}")

    if return_loader:
        torch_dataset = DataLoader(
            dataset,
            batch_size=batch_size,
            drop_last=False,
            shuffle=False,
            collate_fn=(
                dataset.__class__.collate_fn
                if hasattr(dataset.__class__, "collate_fn")
                else None
            ),
        )
        logger.debug(
            f"Loaded `{dataset_name.value}` which contains {len(torch_dataset)} "
            f"batches with a batch size of {batch_size}."
        )
        return dataset, torch_dataset
    else:
        return dataset
