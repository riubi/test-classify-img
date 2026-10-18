import os
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def border_value(img):
    a = np.asarray(img)
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    return int(np.median(border))


def pad_to_square(img):
    w, h = img.size
    s = max(w, h)
    out = Image.new("L", (s, s), border_value(img))
    out.paste(img, ((s - w) // 2, (s - h) // 2))
    return out


def load_image(path):
    img = Image.open(path).convert("L")
    img = pad_to_square(img)
    return img.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)


class RandomRotateBg:
    def __init__(self, small_angle=15, right_angles=True):
        self.small_angle = small_angle
        self.right_angles = right_angles

    def __call__(self, img):
        k = random.randint(0, 3) if self.right_angles else 0
        if k:
            img = img.rotate(90 * k)
        if self.small_angle:
            angle = random.uniform(-self.small_angle, self.small_angle)
            img = TF.rotate(img, angle, interpolation=transforms.InterpolationMode.BILINEAR,
                            fill=border_value(img))
        return img


def get_transforms(aug, mean, std):
    to_tensor = [
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    if aug == "none":
        train = []
    elif aug == "basic":
        train = [
            RandomRotateBg(small_angle=15),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    elif aug == "flip":
        train = [
            RandomRotateBg(small_angle=10, right_angles=False),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    elif aug == "strong":
        train = [
            RandomRotateBg(small_angle=45),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.5, 1.0), ratio=(0.75, 1.33)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4),
        ]
    else:
        raise ValueError(aug)
    return transforms.Compose(train + to_tensor), transforms.Compose(to_tensor)


class PlanktonDataset(Dataset):
    def __init__(self, df, data_dir, class_to_idx, transform=None):
        self.images = [load_image(os.path.join(data_dir, p)) for p in df["path"]]
        self.labels = [class_to_idx[c] for c in df["class"]]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = self.images[i]
        if self.transform:
            img = self.transform(img)
        return img, self.labels[i]


def load_split(path="splits/split.csv"):
    df = pd.read_csv(path)
    classes = sorted(df["class"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    return df, classes, class_to_idx


def dataset_stats(ds):
    x = np.stack([np.asarray(im, dtype=np.float32) / 255.0 for im in ds.images])
    return float(x.mean()), float(x.std())


def seed_worker(_):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)
