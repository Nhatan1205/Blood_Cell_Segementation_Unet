import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp  # Import U-Net from segmentation_models_pytorch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 🎨 Color palette for classes: Background, Blood Cell
COLOR_PALETTE = [
    (0, 0, 0),      # Background (black)
    (255, 255, 255), # Blood Cell (white)
]

def visualize_segmentation_map_binary(image, mask):
    """Generate colored overlay for the original image with a binary mask"""
    image = np.array(image).astype(np.uint8)
    h, w = mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

    # Apply color based on each class
    for class_id, color in enumerate(COLOR_PALETTE):
        colored_mask[mask == class_id] = color

    # Convert to BGR (for OpenCV)
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Overlay the mask onto the image
    overlayed_image = cv2.addWeighted(bgr_image, 0.6, colored_mask, 0.4, 0)

    return overlayed_image, colored_mask

def visualize_prediction_binary_with_accuracy(model_path, image_path, mask_path, save_overlay=False, device=None):
    """Display the original image, true mask, predicted mask, overlay, and accuracy"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔹 Using device: {device}")

    # ✅ Load the pre-trained model from segmentation_models_pytorch
    model = smp.Unet(
        encoder_name="resnet34",  # Encoder type
        encoder_weights="imagenet",  # Pre-trained encoder weights
        in_channels=3,  # RGB images
        classes=1  # Binary segmentation (1 class)
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # ✅ Load the image and true mask
    image = Image.open(image_path).convert("RGB")
    mask_true = Image.open(mask_path).convert("L")  # Mask grayscale
    original_size = image.size

    mask_true_np = np.array(mask_true)
    # Normalize true mask to binary (0 or 1)
    mask_true_binary = (mask_true_np > 0).astype(np.uint8)

    # ✅ Preprocessing
    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    image_np = np.array(image)
    transformed = transform(image=image_np)
    image_tensor = transformed["image"].unsqueeze(0).to(device)

    # ✅ Make prediction
    with torch.no_grad():
        pred_mask_logits = model(image_tensor)  # Output shape: (1, 1, 256, 256)
        pred_mask_probs = torch.sigmoid(pred_mask_logits)
        pred_mask = (pred_mask_probs > 0.5).float().squeeze().cpu().numpy()  # Threshold at 0.5

    # Resize prediction mask to original image size
    pred_mask_resized = Image.fromarray((pred_mask * 255).astype(np.uint8)).resize(original_size, Image.NEAREST)
    pred_mask_resized_np = np.array(pred_mask_resized)
    pred_mask_binary_resized = (pred_mask_resized_np > 127).astype(np.uint8)

    # ✅ Calculate accuracy
    correct_pixels = np.sum(pred_mask_binary_resized == mask_true_binary)
    total_pixels = mask_true_binary.size
    accuracy = correct_pixels / total_pixels

    # ✅ Create colored overlay for prediction and true mask
    overlayed_image, colored_mask = visualize_segmentation_map_binary(image, pred_mask_binary_resized)
    overlayed_true, colored_mask_true = visualize_segmentation_map_binary(image, mask_true_binary)

    # ✅ Display the images
    fig, ax = plt.subplots(1, 4, figsize=(20, 6))

    ax[0].imshow(image_np)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(mask_true_binary, cmap="gray", vmin=0, vmax=1)
    ax[1].set_title("True Mask [0-1]")
    ax[1].axis("off")

    ax[2].imshow(pred_mask_binary_resized, cmap="gray", vmin=0, vmax=1)
    ax[2].set_title("Predicted Mask [0-1]")
    ax[2].axis("off")

    ax[3].imshow(cv2.cvtColor(overlayed_image, cv2.COLOR_BGR2RGB))
    ax[3].set_title("Overlay Prediction")
    ax[3].axis("off")

    plt.suptitle(f"Pixel Accuracy: {accuracy:.4f}", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()

    # ✅ Save the overlay if needed
    if save_overlay:
        filename = os.path.basename(image_path)
        os.makedirs("./assets", exist_ok=True)

        Image.fromarray((pred_mask_binary_resized * 255).astype(np.uint8)).save(f"./assets/{filename[:-4]}_mask_pred.png")
        cv2.imwrite(f"./assets/{filename[:-4]}_mask_color_pred.png", colored_mask)
        cv2.imwrite(f"./assets/{filename[:-4]}_overlay_pred.png", overlayed_image)
        cv2.imwrite(f"./assets/{filename[:-4]}_mask_color_true.png", colored_mask_true)
        cv2.imwrite(f"./assets/{filename[:-4]}_overlay_true.png", overlayed_true)

        print("✅ Saved results in the 'assets' folder!")

    # ✅ Debug min-max values of the mask
    print(f"🎯 True Mask: min={mask_true_binary.min()}, max={mask_true_binary.max()}")
    print(f"🎯 Predicted Mask: min={pred_mask_binary_resized.min()}, max={pred_mask_binary_resized.max()}")
    print(f"🎯 Pixel Accuracy: {accuracy:.4f}")

# 🔥 Call the function for testing
visualize_prediction_binary_with_accuracy(
    model_path='./models/unet_best1.pth',  # Path to your trained model
    image_path='./data/BCCD Dataset with mask/test/original/fe1ee954-ba71-47b3-954a-d20ab940cd7b.png',
    mask_path='./data/BCCD Dataset with mask/test/mask/fe1ee954-ba71-47b3-954a-d20ab940cd7b.png',
    save_overlay=True  # Save the overlay images
)
