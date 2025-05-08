import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from unet import UNet  # Import mô hình UNet
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 🎨 Màu cho từng class: Background, Blood Cell
COLOR_PALETTE = [
    (0, 0, 0),      # Background (đen)
    (255, 255, 255), # Blood Cell (trắng)
]

# def visualize_segmentation_map_binary(image, mask):
#     """Tạo overlay mask màu lên ảnh gốc (cho 2 class)"""
#     image = np.array(image).astype(np.uint8)  # Chuyển ảnh về numpy
#     h, w = mask.shape
#     colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

#     # Áp màu theo từng class
#     for class_id, color in enumerate(COLOR_PALETTE):
#         colored_mask[mask == class_id] = color

#     # Chuyển ảnh về BGR (OpenCV)
#     bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#     # Overlay với ảnh gốc
#     overlayed_image = cv2.addWeighted(bgr_image, 0.6, colored_mask, 0.4, 0)

#     return overlayed_image, colored_mask
def visualize_segmentation_map_binary(image, mask):
    """Tạo overlay mask màu lên ảnh gốc (cho 2 class)"""
    image = np.array(image).astype(np.uint8)  # Chuyển ảnh về numpy
    h, w = mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

    # Áp màu theo từng class
    for class_id, color in enumerate(COLOR_PALETTE):
        colored_mask[mask == class_id] = color

    # Resize the mask to match the image size
    image_resized = cv2.resize(image, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Chuyển ảnh về BGR (OpenCV)
    bgr_image = cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR)

    # Overlay với ảnh gốc
    overlayed_image = cv2.addWeighted(bgr_image, 0.6, colored_mask, 0.4, 0)

    return overlayed_image, colored_mask


def visualize_prediction_binary_with_accuracy(model_path, image_path, mask_path, save_overlay=False, device=None):
    """Hiển thị ảnh gốc, mask thật, mask dự đoán (binary), overlay và accuracy"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔹 Sử dụng thiết bị: {device}")

    # ✅ Load mô hình
    model = UNet(n_channels=3, n_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # ✅ Load ảnh & mask thật
    image = Image.open(image_path).convert("RGB")
    mask_true = Image.open(mask_path).convert("L")  # Mask grayscale
    original_size = image.size

    mask_true_np = np.array(mask_true)
    # Chuẩn hóa mask thật về 0 và 1
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

    # ✅ Dự đoán mask
    with torch.no_grad():
        pred_mask_logits = model(image_tensor)  # Output shape: (1, 1, 256, 256)
        pred_mask_probs = torch.sigmoid(pred_mask_logits)
        pred_mask = (pred_mask_probs > 0.5).float().squeeze().cpu().numpy()  # Ngưỡng 0.5

    # Resize mask về kích thước gốc
    pred_mask_resized = Image.fromarray((pred_mask * 255).astype(np.uint8)).resize(original_size, Image.NEAREST)
    pred_mask_resized_np = np.array(pred_mask_resized)
    pred_mask_binary_resized = (pred_mask_resized_np > 127).astype(np.uint8)

    # ✅ Tính toán Accuracy
    correct_pixels = np.sum(pred_mask_binary_resized == mask_true_binary)
    total_pixels = mask_true_binary.size
    accuracy = correct_pixels / total_pixels

    # ✅ Tạo overlay màu
    overlayed_image, colored_mask = visualize_segmentation_map_binary(image, pred_mask_binary_resized)
    overlayed_true, colored_mask_true = visualize_segmentation_map_binary(image, mask_true_binary)

    # ✅ Hiển thị ảnh gốc, mask thật và mask dự đoán
    fig, ax = plt.subplots(1, 4, figsize=(20, 6))

    ax[0].imshow(image_np)
    ax[0].set_title("Ảnh Gốc")
    ax[0].axis("off")

    ax[1].imshow(mask_true_binary, cmap="gray", vmin=0, vmax=1)
    ax[1].set_title("Mask Thật [0-1]")
    ax[1].axis("off")

    ax[2].imshow(pred_mask_binary_resized, cmap="gray", vmin=0, vmax=1)
    ax[2].set_title("Mask Dự Đoán [0-1]")
    ax[2].axis("off")

    ax[3].imshow(cv2.cvtColor(overlayed_image, cv2.COLOR_BGR2RGB))
    ax[3].set_title("Overlay Dự Đoán")
    ax[3].axis("off")

    plt.suptitle(f"Pixel Accuracy: {accuracy:.4f}", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()

    # ✅ Lưu ảnh nếu cần
    if save_overlay:
        filename = os.path.basename(image_path)
        os.makedirs("./assets", exist_ok=True)  # Tạo thư mục nếu chưa có

        Image.fromarray((pred_mask_binary_resized * 255).astype(np.uint8)).save(f"./assets/{filename[:-4]}_mask_pred.png")
        cv2.imwrite(f"./assets/{filename[:-4]}_mask_color_pred.png", colored_mask)
        cv2.imwrite(f"./assets/{filename[:-4]}_overlay_pred.png", overlayed_image)
        cv2.imwrite(f"./assets/{filename[:-4]}_mask_color_true.png", colored_mask_true)
        cv2.imwrite(f"./assets/{filename[:-4]}_overlay_true.png", overlayed_true)

        print("✅ Lưu kết quả vào thư mục assets!")

    # ✅ Debug min-max mask
    print(f"🎯 Mask Thật: min={mask_true_binary.min()}, max={mask_true_binary.max()}")
    print(f"🎯 Mask Dự Đoán: min={pred_mask_binary_resized.min()}, max={pred_mask_binary_resized.max()}")
    print(f"🎯 Pixel Accuracy: {accuracy:.4f}")

# 🔥 Gọi hàm kiểm tra (đã cập nhật cho 2 class và accuracy)
visualize_prediction_binary_with_accuracy(
    model_path='./models/unet_best-8-5-16-26.pth',
    # image_path='./data/BCCD Dataset with mask/test/original/e11515b4-9527-4c23-a0ba-43719bacca0d.png',
    # mask_path='./data/BCCD Dataset with mask/test/mask/e11515b4-9527-4c23-a0ba-43719bacca0d.png',
    image_path='./data/Neutrophil4.jpg',
    # image_path='./data/KRD-WBC dataset/Dataset/image/image298.jpg',
    mask_path='./data/KRD-WBC dataset/Dataset/mask/mask298.jpg',
    save_overlay=True  # Lưu ảnh overlay vào thư mục assets
)