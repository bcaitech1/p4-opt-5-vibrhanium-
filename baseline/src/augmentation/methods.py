# -*- coding: utf-8 -*-
"""Augmentation methods.

- Author: wlaud1001
- Email: wlaud1001@snu.ac.kr
- Reference:
    https://github.com/j-marple-dev/model_compression
"""

from abc import ABC
import random
from typing import List, Tuple

from PIL.Image import Image

from src.augmentation.transforms import transforms_info


class Augmentation(ABC):
    """Abstract class used by all augmentation methods."""

    def __init__(self, n_level: int = 10) -> None:
        """Initialize."""
        self.transforms_info = transforms_info()
        self.n_level = n_level

    def _apply_augment(self, img: Image, name: str, level: int) -> Image:
        """Apply and get the augmented image.

        Args:
            img (Image): an image to augment
            level (int): magnitude of augmentation in [0, n_level]

        returns:
            Image: an augmented image
        """
        assert 0 <= level <= self.n_level
        augment_fn, low, high = self.transforms_info[name]
        return augment_fn(img.copy(), level * (high - low) / self.n_level + low)


class SequentialAugmentation(Augmentation):
    """Sequential augmentation class."""

    def __init__(
        self,
        policies: List[Tuple[str, float, int]],
        n_level: int = 10,
    ) -> None:
        """Initialize."""
        super().__init__(n_level)
        self.policies = policies

    def __call__(self, img: Image) -> Image:
        """Run augmentations."""
        for name, pr, level in self.policies:
            if random.random() > pr:
                continue
            img = self._apply_augment(img, name, level)
        return img


class RandAugmentation(Augmentation):
    """Random augmentation class.

    References:
        RandAugment: Practical automated data augmentation with a reduced search space
        (https://arxiv.org/abs/1909.13719)

    """

    def __init__(
        self,
        transforms: List[str],
        n_select: int = 2,
        level: int = 14,
        n_level: int = 31,
    ) -> None:
        """Initialize."""
        super().__init__(n_level)
        self.n_select = n_select
        self.level = level if isinstance(level, int) and 0 <= level <= n_level else None
        self.transforms = transforms

    def __call__(self, img: Image) -> Image:
        """Run augmentations."""
        chosen_transforms = random.sample(self.transforms, k=self.n_select)
        for transf in chosen_transforms:
            level = self.level if self.level else random.randint(0, self.n_level)
            img = self._apply_augment(img, transf, level)
        return img
