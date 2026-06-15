"""
test_augmentation.py — Unit tests for augmentation pipeline and CutMix.

Tests:
  1. Mask values stay in valid class range [0, 6] after all augmentations
  2. Image and mask spatial dimensions always match
  3. CutMix produces correct mixed masks
  4. CutMix preserves spatial dimensions
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import yaml

from src.data.augmentation import get_transforms
from src.data.cutmix import CutMix, cutmix_batch, rand_bbox


def load_test_config():
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", "config.yaml"
    )
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_dummy_image_mask(
    H: int = 512, W: int = 512, num_classes: int = 7
) -> tuple:
    """Create a dummy image (HxWx3) and mask (HxW) for testing."""
    image = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    mask = np.random.randint(0, num_classes, (H, W), dtype=np.int64)
    return image, mask


class TestTrainAugmentation:
    """Test that train augmentations preserve valid mask values and shapes."""

    def setup_method(self):
        self.config = load_test_config()
        self.transform = get_transforms("train", self.config)

    def test_mask_values_in_valid_range(self):
        """After all augmentations, mask values must stay in [0, 6]."""
        image, mask = create_dummy_image_mask()
        augmented = self.transform(image=image, mask=mask)
        aug_mask = augmented["mask"]
        assert aug_mask.min() >= 0, f"Mask has negative values: {aug_mask.min()}"
        assert aug_mask.max() <= 6, f"Mask has values > 6: {aug_mask.max()}"

    def test_image_mask_dimensions_match(self):
        """Image and mask must have the same spatial dimensions."""
        image, mask = create_dummy_image_mask()
        augmented = self.transform(image=image, mask=mask)
        aug_image = augmented["image"]
        aug_mask = augmented["mask"]
        assert aug_image.shape[:2] == aug_mask.shape[:2], (
            f"Image shape {aug_image.shape[:2]} != mask shape {aug_mask.shape[:2]}"
        )

    def test_crop_size(self):
        """After RandomCrop, output should be crop_size."""
        crop_size = tuple(self.config["augmentation"]["train"]["crop_size"])
        image, mask = create_dummy_image_mask()
        augmented = self.transform(image=image, mask=mask)
        aug_image = augmented["image"]
        assert aug_image.shape[:2] == crop_size, (
            f"Expected crop {crop_size}, got {aug_image.shape[:2]}"
        )

    def test_normalized_dtype(self):
        """After Normalize, image should be float32 in approximately [-2, 2]."""
        image, mask = create_dummy_image_mask()
        augmented = self.transform(image=image, mask=mask)
        aug_image = augmented["image"]
        assert aug_image.dtype == np.float32, f"Expected float32, got {aug_image.dtype}"
        assert -3 < aug_image.min() < 3, f"Unexpected image min: {aug_image.min()}"
        assert -3 < aug_image.max() < 3, f"Unexpected image max: {aug_image.max()}"


class TestValAugmentation:
    """Test that val augmentation only normalizes."""

    def setup_method(self):
        self.config = load_test_config()
        self.transform = get_transforms("val", self.config)

    def test_val_preserves_spatial_size(self):
        """Val transform should not crop; preserve original size."""
        image, _ = create_dummy_image_mask(H=1024, W=1024)
        augmented = self.transform(image=image)
        assert augmented["image"].shape[:2] == (1024, 1024), (
            f"Val transform changed size: {augmented['image'].shape[:2]}"
        )


class TestCutMix:
    """Test CutMix implementation."""

    def test_cutmix_output_shapes(self):
        """CutMix must preserve batch shapes."""
        B, C, H, W = 4, 3, 512, 512
        images = torch.randn(B, C, H, W)
        masks = torch.randint(0, 7, (B, H, W))

        mixed_images, mixed_masks = cutmix_batch(images, masks, alpha=1.0, prob=1.0)

        assert mixed_images.shape == (B, C, H, W), (
            f"Mixed images shape changed: {mixed_images.shape}"
        )
        assert mixed_masks.shape == (B, H, W), (
            f"Mixed masks shape changed: {mixed_masks.shape}"
        )

    def test_cutmix_mixed_mask_values(self):
        """CutMix mask values must still be in valid class range."""
        B, C, H, W = 4, 3, 512, 512
        images = torch.randn(B, C, H, W)
        masks = torch.randint(0, 7, (B, H, W))

        _, mixed_masks = cutmix_batch(images, masks, alpha=1.0, prob=1.0)

        assert mixed_masks.min() >= 0, f"Negative values in mixed mask"
        assert mixed_masks.max() < 7, f"Values >= 7 in mixed mask"

    def test_cutmix_actual_content_mixed(self):
        """Verify that content is actually mixed between pairs."""
        B, C, H, W = 2, 3, 64, 64

        # Create two distinct images/masks
        images = torch.zeros(B, C, H, W)
        images[0] = 0.0   # All zeros
        images[1] = 1.0   # All ones
        masks = torch.zeros(B, H, W, dtype=torch.long)
        masks[0] = 0       # All class 0
        masks[1] = 3       # All class 3

        # Apply CutMix with prob=1.0, alpha=1.0 on first sample only
        mixed_images, mixed_masks = cutmix_batch(images, masks, alpha=1.0, prob=1.0)

        # The first sample should have content from both samples
        # (since rand_bbox is random, we just verify that the region was replaced)
        mixed_img_0 = mixed_images[0]
        mixed_mask_0 = mixed_masks[0]

        # Some pixels should be from sample 0 (value 0) and some from sample 1 (value 1)
        unique_img_vals = torch.unique(mixed_img_0)
        unique_mask_vals = torch.unique(mixed_mask_0)

        assert len(unique_mask_vals) >= 2, (
            f"Mixed mask should have at least 2 unique values, got {unique_mask_vals}"
        )
        assert 0 in unique_mask_vals and 3 in unique_mask_vals, (
            f"Mixed mask missing expected classes 0 and 3, got {unique_mask_vals}"
        )

    def test_cutmix_no_apply_when_prob_zero(self):
        """When prob=0, CutMix must be a no-op."""
        B, C, H, W = 4, 3, 512, 512
        images = torch.randn(B, C, H, W)
        masks = torch.randint(0, 7, (B, H, W))

        images_copy = images.clone()
        masks_copy = masks.clone()
        mixed_images, mixed_masks = cutmix_batch(images, masks, alpha=1.0, prob=0.0)

        assert torch.allclose(mixed_images, images_copy), (
            "Images changed when prob=0"
        )
        assert torch.equal(mixed_masks, masks_copy), (
            "Masks changed when prob=0"
        )

    def test_cutmix_class_wrapper(self):
        """CutMix class wrapper works as callable."""
        cutmix = CutMix(alpha=1.0, prob=0.5)
        B, C, H, W = 4, 3, 512, 512
        images = torch.randn(B, C, H, W)
        masks = torch.randint(0, 7, (B, H, W))

        mixed_images, mixed_masks = cutmix(images, masks)
        assert mixed_images.shape == images.shape
        assert mixed_masks.shape == masks.shape

    def test_rand_bbox_bounds(self):
        """rand_bbox must return coordinates within image bounds."""
        H, W = 512, 512
        for _ in range(100):
            lam = torch.rand(1).item() * 0.8 + 0.1  # [0.1, 0.9]
            x1, y1, x2, y2 = rand_bbox(H, W, lam)
            assert x1 >= 0 and x2 <= W, f"x bounds violated: ({x1}, {x2})"
            assert y1 >= 0 and y2 <= H, f"y bounds violated: ({y1}, {y2})"
            assert x1 < x2 and y1 < y2, (
                f"Empty box: ({x1}, {x2}), ({y1}, {y2})"
            )
