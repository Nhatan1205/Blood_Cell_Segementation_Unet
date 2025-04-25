# src/dataset.py
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class BloodCellDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(('.png'))])
        self.transform = transform
        print(f"📸 Number of images: {len(self.image_files)}, 🩸 masks: {len(self.mask_files)}")

    def __len__(self):
        return min(len(self.image_files), len(self.mask_files))

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Convert mask to grayscale
        mask = np.array(mask, dtype=np.float32) / 255.0  # Normalize to 0-1

        if self.transform:
            transformed = self.transform(image=np.array(image), mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        return image, mask.unsqueeze(0)  # Add channel dimension to match U-Net output
