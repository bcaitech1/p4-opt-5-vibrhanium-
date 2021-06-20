import os
import os.path as p

import cv2
from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .augmentations import get_augmentations


# -- Dataset/Dataloader
class CustomDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        super().__init__()
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        label = self.labels[idx]
        file_path = self.file_paths[idx]

        # Read an image with OpenCV
        image = cv2.imread(file_path)

        # By default OpenCV uses BGR color space for color images,
        # so we need to convert the image to RGB color space.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, label


def get_data_paths(dir_path):
    img_paths = glob(p.join(dir_path, "*"))
    img_paths.sort()

    result_paths = []
    result_labels = []
    for idx, img_path in enumerate(img_paths):
        paths = glob(p.join(img_path, "*"))
        result_paths.extend(paths)
        result_labels.extend([idx] * len(paths))

    return result_paths, result_labels


def get_dataloader(data_path, batch_size, img_size=224):
    transform_train, transform_test = get_augmentations(img_size=img_size)

    train_paths, train_labels = get_data_paths(p.join(data_path, "train"))
    train_dataset = CustomDataset(train_paths, train_labels, transform=transform_train)
    train_loader = DataLoader(
        dataset=train_dataset,
        pin_memory=(torch.cuda.is_available()),
        shuffle=True,
        batch_size=batch_size,
        num_workers=8,
        drop_last=True,
    )

    valid_paths, valid_labels = get_data_paths(p.join(data_path, "val"))
    valid_dataset = CustomDataset(valid_paths, valid_labels, transform=transform_test)
    valid_loader = DataLoader(
        dataset=valid_dataset,
        pin_memory=(torch.cuda.is_available()),
        shuffle=False,
        batch_size=batch_size,
        num_workers=4,
    )

    return train_loader, valid_loader


# -- Loss function
class F1CELoss(nn.Module):
    def __init__(self, classes=8, epsilon=1e-7):
        super().__init__()
        self.classes = classes
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1

        ce_loss = nn.functional.cross_entropy(y_pred, y_true)

        y_true = F.one_hot(y_true, self.classes).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)

        f1_loss = 1 - f1.mean()

        return f1_loss + ce_loss


# -- Other utils
def save_model(model, path):
    """save model to torch script, onnx."""
    try:
        torch.save(model, f=path)
    except:
        print("Failed to save torch")
