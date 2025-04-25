# src/train.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from unet import UNet
from dataset import BloodCellDataset
from utils import get_transforms
from config import image_dir, mask_dir, batch_size, num_epochs, learning_rate

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥 Using device: {device}")

# DataLoader setup
transform = get_transforms()
dataset = BloodCellDataset(image_dir, mask_dir, transform=transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Model, Loss, Optimizer
model = UNet(n_channels=3, n_classes=1).to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# Scheduler for learning rate
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# Training loop
best_val_loss = float("inf")

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]")

    for images, masks in loop:
        images = images.to(device)
        masks = masks.to(device).float()

        optimizer.zero_grad()
        outputs = model(images)
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
        torch.save(model.state_dict(), './models/unet_best.pth')
        print("✅ Model saved with lower validation loss!")

print("🎉 Finished Training")
