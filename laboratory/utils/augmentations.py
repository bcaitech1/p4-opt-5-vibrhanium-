import numpy as np
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform

class SquarePad(ImageOnlyTransform):
    """Square pad to make torch resize to keep aspect ratio."""
    
    def __init__(self, always_apply=False, p=0.5):
        super(SquarePad, self).__init__(always_apply, p)
        
    def apply(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 0, "constant")
    
    def get_params(self):
        return None

    def get_transform_init_args_names(self):
        return None


def get_augmentations(img_size):
    transform_train = A.Compose([
        SquarePad(always_apply=True, p=1.0),
        A.Resize(img_size, img_size),
        A.OneOf([A.RandomRotate90(p=.25),
                A.VerticalFlip(p=.25)], p=1),
        A.CoarseDropout(max_holes=10, max_height=4, max_width=4, p=.25),
        A.ShiftScaleRotate(p=.25),
        A.Normalize(),
        ToTensorV2()
    ])

    transform_test = A.Compose([
        SquarePad(always_apply=True, p=1.0),
        A.Resize(img_size, img_size),
        A.Normalize(),
        ToTensorV2()
    ])

    return transform_train, transform_test