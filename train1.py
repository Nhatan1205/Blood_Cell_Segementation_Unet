import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import segmentation_models_pytorch as smp  # Import the library
import os
from PIL import Image
import numpy as np


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Dataset class (remains the same)
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

        return image, mask.unsqueeze(0)  # Add channel dimension to the mask

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥 Using device: {device}")

# Dataset paths
image_dir = './BCCD Dataset with mask/train/original'
mask_dir = './BCCD Dataset with mask/train/mask'

# Augmentation
transform = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomBrightnessContrast(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
], additional_targets={'mask': 'mask'})

# Dataset & DataLoader
dataset = BloodCellDataset(image_dir, mask_dir, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)

# Initialize pre-built U-Net model from segmentation_models_pytorch
model = smp.Unet(
    encoder_name="resnet34",  # You can choose from several encoders such as 'resnet34', 'resnet50', etc.
    encoder_weights="imagenet",  # Use ImageNet pre-trained weights
    in_channels=3,  # 3 input channels (RGB images)
    classes=1,  # Output 1 channel (binary segmentation)
).to(device)

# Loss function & Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4)

# Scheduler to reduce LR if val_loss doesn't improve
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)


# Train model
num_epochs = 15
best_val_loss = float("inf")

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]")

    for images, masks in loop:
        images = images.to(device)
        masks = masks.to(device).float()

        optimizer.zero_grad()
        outputs = model(images)  # No need to apply sigmoid here!
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    train_loss /= len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]"):
            images = images.to(device)
            masks = masks.to(device).float()

            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    scheduler.step(val_loss)

    print(f"🔵 Train Loss: {train_loss:.4f} | 🟢 Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), './unet_best1.pth')
        print("✅ Model saved with lower validation loss!")

print("🎉 Finished Training")
