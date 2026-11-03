import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from src.dataloader.data_pack import CameraCreator


class CameraAlphaTests(unittest.TestCase):
    def create(self, image, mask=None, **options):
        with patch('src.dataloader.data_pack.Camera', side_effect=lambda **kw: kw):
            return CameraCreator(**options)(image, np.eye(4), 1., 1., mask=mask)

    def test_rgba_alpha_stays_normalized_and_blends_white(self):
        rgba = np.zeros((8, 8, 4), dtype=np.uint8)
        rgba[:, :, 0] = 255
        rgba[:, :, 3] = 128
        result = self.create(Image.fromarray(rgba), alpha_is_white=True)
        torch.testing.assert_close(result['mask'], torch.full((1, 8, 8), 128 / 255))
        torch.testing.assert_close(result['image'][0], torch.ones(8, 8))
        torch.testing.assert_close(result['image'][1], torch.full((8, 8), 127 / 255))

    def test_resize_and_external_mask_combine_with_alpha(self):
        rgba = np.full((8, 8, 4), 128, dtype=np.uint8)
        result = self.create(Image.fromarray(rgba),
                             Image.fromarray(np.full((2, 2), 255, dtype=np.uint8)),
                             res_downscale=2, skip_blend_alpha=True)
        self.assertEqual(result['image'].shape, (3, 4, 4))
        torch.testing.assert_close(result['mask'], torch.full((1, 4, 4), 128 / 255))

    def test_rgb_external_mask_and_absent_mask(self):
        image = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
        self.assertIsNone(self.create(image)['mask'])
        result = self.create(image, Image.fromarray(np.full((2, 2), 255, dtype=np.uint8)))
        torch.testing.assert_close(result['mask'], torch.ones(1, 8, 8))
