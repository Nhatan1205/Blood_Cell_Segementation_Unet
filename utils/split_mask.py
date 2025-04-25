import os
import shutil

def split_masks(image_sets_dir, masks_dir, output_train_dir, output_val_dir):
    """
    Phân chia các ảnh mask vào thư mục train và val dựa trên file ImageSets.

    Args:
        image_sets_dir (str): Đường dẫn đến thư mục ImageSets/Main.
        masks_dir (str): Đường dẫn đến thư mục chứa tất cả các ảnh mask.
        output_train_dir (str): Đường dẫn đến thư mục để lưu mask train.
        output_val_dir (str): Đường dẫn đến thư mục để lưu mask val.
    """
    os.makedirs(output_train_dir, exist_ok=True)
    os.makedirs(output_val_dir, exist_ok=True)

    for subset in ['train', 'val']:
        image_list_file = os.path.join(image_sets_dir, f'{subset}.txt')
        with open(image_list_file, 'r') as f:
            image_names = [line.strip() for line in f]

        for image_name in image_names:
            mask_name = f'{image_name}_mask.png'  # Giả sử hậu tố mask là _mask.png
            src_mask_path = os.path.join(masks_dir, mask_name)

            if os.path.exists(src_mask_path):
                if subset == 'train':
                    dst_mask_path = os.path.join(output_train_dir, mask_name)
                else:
                    dst_mask_path = os.path.join(output_val_dir, mask_name)
                shutil.move(src_mask_path, dst_mask_path)
                print(f"Đã chuyển {mask_name} vào thư mục {subset}")
            else:
                print(f"Warning: Không tìm thấy mask cho ảnh {image_name}")

if __name__ == "__main__":
    image_sets_dir = r"D:\NCKH\BCCD_Dataset\BCCD\ImageSets\Main"
    masks_dir = r"D:\NCKH\TestUnet\masks"
    output_train_dir = r"D:\NCKH\TestUnet\masks\train"
    output_val_dir = r"D:\NCKH\TestUnet\masks\val"

    split_masks(image_sets_dir, masks_dir, output_train_dir, output_val_dir)
    print("Hoàn thành việc phân chia mask.")