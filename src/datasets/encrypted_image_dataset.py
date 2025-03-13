import io
import warnings
from pathlib import Path
from typing import List, Tuple, Union

import torch
from cryptography.fernet import Fernet, InvalidToken, MultiFernet
from PIL import Image, ImageFile, UnidentifiedImageError
from tqdm.auto import tqdm

from ...src.datasets.generic_image_dataset import GenericImageDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class EncryptedImageDataset(GenericImageDataset):
    def __init__(
        self,
        *args,
        enc_keys: Union[List[str], None],
        cache_in_memory: bool = False,
        val_transform=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # config for selecting the correct transform
        self.cache_in_memory = cache_in_memory
        self.training = True
        self.val_transform = val_transform
        # load all encryption keys
        self.f_keys = []
        self.enc_keys = enc_keys
        if enc_keys is not None:
            # loop over all the keys
            for enc_key_path in enc_keys:
                enc_key_path = Path(enc_key_path)
                if not enc_key_path.exists():
                    raise ValueError("Encryption key does not exist.")
                with open(enc_key_path, "rb") as kf:
                    self.f_keys.append(Fernet(kf.read()))
            # create our multi decryption model
            self.multi_fernet = MultiFernet(self.f_keys)
        # cache the dataset in memory if requested
        if self.cache_in_memory:
            self.cache_samples = []
            self.cache_target = []
            self.cache_path = []
            for idx in tqdm(
                range(len(self)),
                total=len(self),
                desc="Loading Dataset to Memory",
            ):
                entry = self.__getitem__(index=idx, load_from_cache=False)
                if self.return_path:
                    sample, path, target = entry
                else:
                    sample, target = entry
                    path = None
                self.cache_samples.append(sample)
                self.cache_target.append(target)
                self.cache_path.append(path)

    def load_encrypted_image(self, path: str):
        try:
            # first try to load the image without encryption,
            # we'll do this first, since it is more often the case
            # and thus more efficient
            image = Image.open(path)
            image = image.convert("RGB")
            return image
        except UnidentifiedImageError:
            with open(path, "rb") as f:
                # if this exception is thrown, the file is encrypted
                # decrypt encrypted image
                try:
                    dec_img = self.multi_fernet.decrypt(f.read())
                except InvalidToken:
                    # if this error is thrown we don't have the correct key
                    # to decrypt the image
                    warnings.warn(
                        f"No valid key available to decrypt the image: {path}"
                    )
                    return None
                # check if the decoded byte string is not empty
                if dec_img.rstrip(b"\x00") != b"":
                    # get PIL image from bytes
                    return Image.open(io.BytesIO(dec_img)).convert("RGB")
                else:
                    warnings.warn(
                        f"Image {path} is an encrypted but EMPTY image"
                        ", please remove it."
                    )
                    return None

    def __getitem__(self, index, load_from_cache: bool = True) -> Union[
        Tuple[Union[torch.Tensor, Image.Image], int],
        Tuple[Union[torch.Tensor, Image.Image], str, int],
        None,
    ]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        # load the sample from memory if we've loaded it
        if self.cache_in_memory and load_from_cache:
            sample = self.cache_samples[index]
            target = self.cache_target[index]
            return sample, target

        img_path = self.meta_data.loc[self.meta_data.index[index], self.IMG_COL]
        # check if encryption should be used or not
        if self.enc_keys is not None:
            # use our custom encrypted loader
            image = self.load_encrypted_image(img_path)
        else:
            # without a key use the regular loader from pytorch
            image = Image.open(img_path)
            image = image.convert("RGB")

        # return None if loading failed
        if image is None:
            return None

        if self.transform and self.training:
            image = self.transform(image)
        elif self.val_transform and not self.training:
            image = self.val_transform(image)

        diagnosis = self.meta_data.loc[self.meta_data.index[index], self.LBL_COL]
        if self.return_path:
            return image, img_path, int(diagnosis)
        else:
            return image, int(diagnosis)

    @staticmethod
    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)
