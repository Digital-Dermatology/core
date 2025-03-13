import unittest

import torch
from torchvision import transforms

from src.augmentations.simclr import SimCLRDataAugmentation


class TestSimclrAugmentation(unittest.TestCase):
    def test_one_augmentation(self):
        target_shape = (3, 224, 224)
        rand_img = transforms.ToPILImage()(torch.rand((3, 512, 512)))
        simclr_aug = SimCLRDataAugmentation(
            target_size=target_shape[-1],
            two_augmentations=False,
        )

        aug_1, aug_2 = simclr_aug(rand_img)

        self.assertEqual(type(aug_1), torch.Tensor)
        self.assertEqual(type(aug_2), torch.Tensor)
        self.assertIsNotNone(aug_1)
        self.assertIsNotNone(aug_2)
        self.assertFalse(torch.equal(aug_1, aug_2))
        self.assertEqual(aug_1.shape, target_shape)
        self.assertEqual(aug_2.shape, target_shape)

    def test_two_augmentation(self):
        target_shape = (3, 224, 224)
        rand_img = transforms.ToPILImage()(torch.rand((3, 512, 512)))
        simclr_aug = SimCLRDataAugmentation(
            target_size=target_shape[-1],
            two_augmentations=True,
        )

        aug_1, aug_2 = simclr_aug(rand_img)

        self.assertEqual(type(aug_1), torch.Tensor)
        self.assertEqual(type(aug_2), torch.Tensor)
        self.assertIsNotNone(aug_1)
        self.assertIsNotNone(aug_2)
        self.assertFalse(torch.equal(aug_1, aug_2))
        self.assertEqual(aug_1.shape, target_shape)
        self.assertEqual(aug_2.shape, target_shape)


if __name__ == "__main__":
    unittest.main()
