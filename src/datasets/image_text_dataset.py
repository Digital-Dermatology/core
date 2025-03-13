import random
import string
from typing import List, Tuple

from clip import tokenize
from PIL import Image
from torch.utils.data import Dataset

from ...src.datasets.base_dataset import BaseDataset


class ImageTextDataset(Dataset):

    def __init__(
        self,
        dataset: BaseDataset,
        label_templates: List[str],
        template_key: str = "label",
    ):
        super().__init__()
        self.dataset = dataset
        self.label_templates = label_templates
        self.template_key = template_key
        check_templates = all(
            [
                self.template_key in ImageTextDataset.check_string_format_arguments(x)
                for x in self.label_templates
            ]
        )
        if not check_templates:
            raise ValueError(
                f"Label templates do not all have the template key: {template_key}"
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> Tuple[Image.Image, str]:
        image, label = self.dataset.__getitem__(index=index)
        label = self.dataset.classes[label]
        template = random.choice(self.label_templates)
        label_text = template.format(**{self.template_key: label}).lower().strip()
        return image, tokenize(label_text)[0]

    @staticmethod
    def check_string_format_arguments(string_to_check):
        return [
            tup[1]
            for tup in string.Formatter().parse(string_to_check)
            if tup[1] is not None
        ]
