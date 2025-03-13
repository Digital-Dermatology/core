from abc import ABC, abstractmethod

from torch.utils.data import Dataset
from torchvision import transforms


class DatasetWrapper(ABC, Dataset):
    def __init__(self, ds: Dataset):
        super().__init__()
        self.ds = ds
        self.preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    @abstractmethod
    def __getitem_internal__(self, idx, preprocess=True):
        pass

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.__getitem_internal__(idx, True)

    def raw(self, idx):
        return self.__getitem_internal__(idx, False)
