import io
import shutil
import tempfile
import time
import unittest
from pathlib import Path

import PIL
from cryptography.fernet import Fernet
from PIL import Image

from src.datasets.encrypted_image_dataset import EncryptedImageDataset


class TestEncryptedFolder(unittest.TestCase):
    def setUp(self):
        self.example_img_path = Path("tests/test_utils/imgs/sweet-syndrome40.jpg")
        self.example_img = Image.open(self.example_img_path)
        # create temp image folder
        self.create_dummy_files()
        # start the test timer
        self.startTime = time.time()

    def create_dummy_files(self):
        # create encrypted temp image folder
        self.tmp_dir = Path(tempfile.mkdtemp())
        (self.tmp_dir / "cls").mkdir(parents=True, exist_ok=True)
        # create decrypted temp image folder
        self.tmp_dec_dir = Path(tempfile.mkdtemp())
        (self.tmp_dec_dir / "cls").mkdir(parents=True, exist_ok=True)
        # store dummy key
        key = Fernet.generate_key()
        with open(f"{self.tmp_dir}/key", "wb") as kf:
            kf.write(key)
        f_key = Fernet(key)
        # store some more random keys
        self.rand_keys = []
        for i in range(3):
            key = Fernet.generate_key()
            key_path = f"{self.tmp_dir}/key_{i}"
            self.rand_keys.append(key_path)
            with open(key_path, "wb") as kf:
                kf.write(key)
        # create encrypted images
        for i in range(100):
            im_byte_arr = io.BytesIO()
            self.example_img.save(im_byte_arr, format="JPEG")
            self.example_img.save(f"{self.tmp_dec_dir}/cls/img_{i}.jpg")
            enc_byte_arr = f_key.encrypt(im_byte_arr.getvalue())
            with open(f"{self.tmp_dir}/cls/enc_img_{i}.jpg", "wb") as writer:
                writer.write(enc_byte_arr)

    def tearDown(self):
        # end the timer
        t = time.time() - self.startTime
        print("%s: %.3f" % (self.id(), t))
        # remove all utility files
        shutil.rmtree(self.tmp_dir)
        shutil.rmtree(self.tmp_dec_dir)

    def test_decrypting_images(self):
        dataset = EncryptedImageDataset(
            self.tmp_dir, enc_keys=[str(self.tmp_dir / "key")]
        )
        for img, _ in dataset:
            self.assertEqual(type(img), Image.Image)
            self.assertIsNotNone(img)

    def test_decrypting_images_caching(self):
        dataset = EncryptedImageDataset(
            self.tmp_dir,
            enc_keys=[str(self.tmp_dir / "key")],
            cache_in_memory=True,
        )
        for img, _ in dataset:
            self.assertEqual(type(img), Image.Image)
            self.assertIsNotNone(img)

    def test_decrypting_images_multiple_keys(self):
        dataset = EncryptedImageDataset(
            self.tmp_dir, enc_keys=self.rand_keys + [str(self.tmp_dir / "key")]
        )
        for img, _ in dataset:
            self.assertEqual(type(img), Image.Image)
            self.assertIsNotNone(img)

    def test_loading_not_encrypted_images(self):
        dataset = EncryptedImageDataset(self.tmp_dec_dir, enc_keys=None)
        for img, _ in dataset:
            self.assertEqual(type(img), Image.Image)
            self.assertIsNotNone(img)

    def test_decrypting_non_encrypted_images(self):
        dataset = EncryptedImageDataset(
            self.tmp_dec_dir, enc_keys=[str(self.tmp_dir / "key")]
        )
        for img, _ in dataset:
            self.assertEqual(type(img), Image.Image)
            self.assertIsNotNone(img)

    def test_loading_encrypted_images(self):
        with self.assertRaises(PIL.UnidentifiedImageError):
            dataset = EncryptedImageDataset(self.tmp_dir, enc_keys=None)
            _ = dataset[0]


if __name__ == "__main__":
    unittest.main(verbosity=0)
