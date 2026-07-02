import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class FreuidDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, is_test=False):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        if 'image_path' in row:
            path = row['image_path']
            if path.startswith('train/'):
                img_path = os.path.join(self.img_dir, 'train', path)
            else:
                img_path = os.path.join(self.img_dir, path)
        else:
            # Fallback for test set (specially case of pvt) where image_path might not be present but id is
            img_path = os.path.join(self.img_dir, 'public_test', 'public_test', f"{row['id']}.jpeg")
            if not os.path.exists(img_path):
                img_path = os.path.join(self.img_dir, 'public_test', f"{row['id']}.jpeg")
        
        image = cv2.imread(img_path)
        if image is None:
            # Fallback for missing private test images or corruption
            image = np.zeros((384, 384, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        if self.is_test:
            return image, row['id']
        else:
            label = int(row['label'])
            return image, torch.tensor(label, dtype=torch.float32)

def fxn_get_transforms(img_size=384, is_train=True):

    # Use: Returns train or validation albumentations transform pipelines.
    #      Includes heavy degradations to simulate the analog hole (print-and-capture).

    if is_train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.Affine(translate_percent=(-0.05, 0.05), scale=(0.9, 1.1), rotate=(-10, 10), p=0.5),
            
            # Simulate the Analog Hole (Print-and-Capture)
            A.OneOf([
                A.ImageCompression(quality_range=(50, 90), p=1.0),
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MotionBlur(blur_limit=5, p=1.0),
            ], p=0.2),
            
            # Simulate bad camera conditions
            A.OneOf([
                A.GaussNoise(std_range=(10.0, 50.0), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            ], p=0.15),
            
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
            
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ])
