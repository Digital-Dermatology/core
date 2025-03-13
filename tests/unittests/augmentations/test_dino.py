import unittest

import torch
from torchvision import transforms

from src.augmentations.multi_crop import DINODataAugmentation


class TestDINOAugmentation(unittest.TestCase):
    def test_augmentation(self):
        global_crops_scale = (0.4, 1.0)
        local_crops_scale = (0.05, 0.4)
        global_crops_number = 2
        local_crops_number = 4
        local_shape = (3, 96, 96)
        global_shape = (3, 224, 224)

        rand_img = transforms.ToPILImage()(torch.rand((3, 512, 512)))
        dino_aug = DINODataAugmentation(
            local_crops_number=local_crops_number,
            global_crops_scale=global_crops_scale,
            local_crops_scale=local_crops_scale,
        )

        dino_crops = dino_aug(rand_img)

        global_crops = dino_crops[:global_crops_number]
        local_crops = dino_crops[global_crops_number:]

        self.assertEqual(len(global_crops), global_crops_number)
        self.assertEqual(len(local_crops), local_crops_number)

        for crop in dino_crops:
            self.assertEqual(type(crop), torch.Tensor)
            self.assertIsNotNone(crop)
        for l_crop in local_crops:
            self.assertEqual(l_crop.shape, local_shape)
        for g_crop in global_crops:
            self.assertEqual(g_crop.shape, global_shape)


if __name__ == "__main__":
    unittest.main()
