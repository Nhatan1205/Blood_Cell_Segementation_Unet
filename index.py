import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, 
                             QVBoxLayout, QHBoxLayout, QFileDialog, QLabel, 
                             QProgressBar, QGroupBox, QRadioButton, QButtonGroup, 
                             QSlider, QComboBox)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor

from PyQt5.QtCore import Qt, QSize, QRect
from PyQt5 import QtCore
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from unet import UNet  # Import mô hình UNet
from PIL import Image

# Định nghĩa COLOR_PALETTE và hàm visualize_segmentation_map_binary
COLOR_PALETTE = [
    (0, 0, 0),      # Background (đen)
    (255, 255, 255) # Blood Cell (trắng)
]

class BloodCellSegmentationApp(QMainWindow):
    # Thêm COLOR_PALETTE từ evaluate.py
    COLOR_PALETTE = [
        (0, 0, 0),      # Background (đen)
        (255, 255, 255) # Blood Cell (trắng)
    ]
    
    def __init__(self):
        super().__init__()
        
        # Initialize all attributes first
        self.original_image = None
        self.current_image = None
        self.processed_image = None
        self.current_step = 0
        self.max_steps = 5  # Total number of steps in our pipeline
        
        # Now initialize the UI
        self.initUI()
        
    def initUI(self):
        # Set window properties
        self.setWindowTitle("Blood Cell Segmentation")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("background-color: white; color: #333;")
        
        # Create main widget and layout
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        
        # Create progress bar (top)
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #2196F3;
                border-radius: 5px;
                text-align: center;
                background-color: white;
                height: 25px;
            }
            QProgressBar::chunk {
                background-color: #2196F3;
            }
        """)
        self.progress_bar.setMaximum(self.max_steps)
        self.progress_bar.setValue(0)
        
        # Tạo danh sách tên các bước
        self.step_names = ["Not Started", "Image Acquisition", "Image Segmentation", 
                        "Image Enhancement", "Morphological Operations", "Image Compression"]

        # Thiết lập định dạng hiển thị ban đầu
        self.progress_bar.setFormat(self.step_names[0])

        
        # Create sidebar (left, 30% width)
        sidebar_widget = QWidget()
        sidebar_widget.setMaximumWidth(int(self.width() * 0.3))
        sidebar_widget.setStyleSheet("background-color: #f0f8ff;")
        sidebar_layout = QVBoxLayout()
        
        # Add basic buttons to sidebar
  
        
        self.reset_btn = QPushButton("Reset Steps")
        self.reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border-radius: 5px;
                padding: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
        """)
        self.reset_btn.clicked.connect(self.reset_steps)
        
        # Step 1: Image acquisition
        self.acquisition_group = QGroupBox("Step 1: Acquisition")
        self.acquisition_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        acquisition_layout = QVBoxLayout()
        self.open_btn = QPushButton("Open Image")
        self.open_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border-radius: 5px;
                padding: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
        """)
        self.open_btn.clicked.connect(self.open_image)
        acquisition_layout.addWidget(self.open_btn)
        self.acquisition_group.setLayout(acquisition_layout)
        
        # Add processing step buttons
        self.step_buttons = []
        
        # Step 2: Image Segmentation
        self.segmentation_group = QGroupBox("Step 2: Image Segmentation")
        self.segmentation_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        segmentation_layout = QVBoxLayout()
        
        self.seg_unet_btn = QRadioButton("U-Net")
        self.seg_threshold_btn = QRadioButton("Thresholding")
        self.seg_btn_group = QButtonGroup()
        self.seg_btn_group.addButton(self.seg_unet_btn)
        self.seg_btn_group.addButton(self.seg_threshold_btn)
        
        self.apply_segmentation_btn = QPushButton("Apply Segmentation")
        self.apply_segmentation_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border-radius: 5px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
        """)
        self.apply_segmentation_btn.clicked.connect(self.apply_segmentation)
        
        segmentation_layout.addWidget(self.seg_unet_btn)
        segmentation_layout.addWidget(self.seg_threshold_btn)
        segmentation_layout.addWidget(self.apply_segmentation_btn)
        self.segmentation_group.setLayout(segmentation_layout)
        
        # Step 3: Image Enhancement
        self.enhancement_group = QGroupBox("Step 3: Image Enhancement")
        self.enhancement_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        enhancement_layout = QVBoxLayout()
        
        self.enhance_hist_btn = QRadioButton("Histogram Equalization")
        self.enhance_median_btn = QRadioButton("Median Filter")
        self.enhance_btn_group = QButtonGroup()
        self.enhance_btn_group.addButton(self.enhance_hist_btn)
        self.enhance_btn_group.addButton(self.enhance_median_btn)
        
        self.apply_enhancement_btn = QPushButton("Apply Enhancement")
        self.apply_enhancement_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border-radius: 5px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
        """)
        self.apply_enhancement_btn.clicked.connect(self.apply_enhancement)
        
        enhancement_layout.addWidget(self.enhance_hist_btn)
        enhancement_layout.addWidget(self.enhance_median_btn)
        enhancement_layout.addWidget(self.apply_enhancement_btn)
        self.enhancement_group.setLayout(enhancement_layout)
        
        # Step 4: Morphological Operations
        self.morphology_group = QGroupBox("Step 4: Morphological Operations")
        self.morphology_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        morphology_layout = QVBoxLayout()
        
        self.morph_combo = QComboBox()
        self.morph_combo.addItems(["Erosion", "Dilation", "Opening", "Closing"])
        self.morph_combo.setStyleSheet("padding: 5px;")
        
        self.kernel_size_slider = QSlider(Qt.Horizontal)
        self.kernel_size_slider.setMinimum(1)
        self.kernel_size_slider.setMaximum(15)
        self.kernel_size_slider.setValue(3)
        self.kernel_size_slider.setTickPosition(QSlider.TicksBelow)
        self.kernel_size_slider.setTickInterval(2)
        
        self.kernel_size_label = QLabel(f"Kernel Size: {self.kernel_size_slider.value()}")
        self.kernel_size_slider.valueChanged.connect(
            lambda v: self.kernel_size_label.setText(f"Kernel Size: {v}")
        )
        
        self.apply_morphology_btn = QPushButton("Apply Morphology")
        self.apply_morphology_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border-radius: 5px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
        """)
        self.apply_morphology_btn.clicked.connect(self.apply_morphology)
        
        morphology_layout.addWidget(self.morph_combo)
        morphology_layout.addWidget(self.kernel_size_label)
        morphology_layout.addWidget(self.kernel_size_slider)
        morphology_layout.addWidget(self.apply_morphology_btn)
        self.morphology_group.setLayout(morphology_layout)
        
        # Step 5: Image Compression
        self.compression_group = QGroupBox("Step 5: Image Compression")
        self.compression_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        compression_layout = QVBoxLayout()
        
        self.compression_slider = QSlider(Qt.Horizontal)
        self.compression_slider.setMinimum(0)
        self.compression_slider.setMaximum(100)
        self.compression_slider.setValue(80)
        self.compression_slider.setTickPosition(QSlider.TicksBelow)
        self.compression_slider.setTickInterval(10)
        
        self.compression_label = QLabel(f"Quality: {self.compression_slider.value()}%")
        self.compression_slider.valueChanged.connect(
            lambda v: self.compression_label.setText(f"Quality: {v}%")
        )
        
        self.apply_compression_btn = QPushButton("Apply Compression")
        self.apply_compression_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border-radius: 5px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
        """)
        self.apply_compression_btn.clicked.connect(self.apply_compression)
        
        compression_layout.addWidget(self.compression_label)
        compression_layout.addWidget(self.compression_slider)
        compression_layout.addWidget(self.apply_compression_btn)
        self.compression_group.setLayout(compression_layout)
        
        # Add save button
        self.save_btn = QPushButton("Save Processed Image")
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
                padding: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.save_btn.clicked.connect(self.save_image)
        
        # Add all widgets to sidebar
        sidebar_layout.addWidget(self.reset_btn)
        sidebar_layout.addWidget(self.acquisition_group)
        sidebar_layout.addWidget(self.segmentation_group)
        sidebar_layout.addWidget(self.enhancement_group)
        sidebar_layout.addWidget(self.morphology_group)
        sidebar_layout.addWidget(self.compression_group)
        sidebar_layout.addWidget(self.save_btn)
        sidebar_layout.addStretch()
        
        sidebar_widget.setLayout(sidebar_layout)
        
        # Create main window (right, 70% width)
        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px dashed #ccc; font-size: 18px;")
        
        # Combine layouts
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.progress_bar)
        right_layout.addWidget(self.image_label)
        
        main_layout.addWidget(sidebar_widget)
        
        right_widget = QWidget()
        right_widget.setLayout(right_layout)
        main_layout.addWidget(right_widget)
        
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # Disable processing buttons initially
        self.display_project_info()
        self.update_ui_state()
    
    def update_ui_state(self):
        has_image = self.original_image is not None
        self.reset_btn.setEnabled(has_image)
        self.segmentation_group.setEnabled(has_image and self.current_step >= 1)
        self.enhancement_group.setEnabled(has_image and self.current_step >= 2)
        self.morphology_group.setEnabled(has_image and self.current_step >= 3)
        self.compression_group.setEnabled(has_image and self.current_step >= 4)
        self.save_btn.setEnabled(has_image and self.current_step > 0)
    
    def open_image(self):
        self.current_step = 1  # Image acquisition complete
        self.progress_bar.setValue(self.current_step)
        self.progress_bar.setFormat(self.step_names[self.current_step])
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        
        if file_path:
            self.original_image = cv2.imread(file_path)
            self.current_image = self.original_image.copy()
            self.current_step = 1  # Image acquisition complete
            self.progress_bar.setValue(self.current_step)
            self.display_image(self.current_image)
            self.update_ui_state()
    
    def reset_steps(self):
        self.current_step = 1  # Reset to after image acquisition
        self.progress_bar.setValue(self.current_step)
        self.progress_bar.setFormat(self.step_names[self.current_step])

        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self.current_step = 1  # Reset to after image acquisition
            self.progress_bar.setValue(self.current_step)
            self.display_image(self.current_image)
            self.update_ui_state()
    
    def visualize_segmentation_map_binary(self, image, mask):
        """Tạo overlay mask màu lên ảnh gốc (cho 2 class)"""
        # Định nghĩa COLOR_PALETTE
        COLOR_PALETTE = [
            (0, 0, 0),      # Background (đen)
            (255, 255, 255) # Blood Cell (trắng)
        ]
        
        image = np.array(image).astype(np.uint8)  # Chuyển ảnh về numpy
        h, w = mask.shape
        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Áp màu theo từng class
        for class_id, color in enumerate(COLOR_PALETTE):
            colored_mask[mask == class_id] = color
            
        # Chuyển ảnh về BGR (OpenCV)
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Overlay với ảnh gốc
        overlayed_image = cv2.addWeighted(bgr_image, 0.6, colored_mask, 0.4, 0)
        
        return overlayed_image, colored_mask

    def apply_segmentation(self):
        self.current_step = 2  # Segmentation complete
        self.progress_bar.setValue(self.current_step)
        self.progress_bar.setFormat(self.step_names[self.current_step])
        
        if self.current_image is None:
            return
            
        if self.seg_threshold_btn.isChecked():
            # Apply thresholding
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            _, segmented = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            self.current_image = cv2.cvtColor(segmented, cv2.COLOR_GRAY2BGR)
            self.display_image(self.current_image)
            
        elif self.seg_unet_btn.isChecked():
            # Sử dụng mô hình U-Net đã được huấn luyện
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Lưu ảnh hiện tại tạm thời để xử lý
            temp_image_path = "temp_image.png"
            cv2.imwrite(temp_image_path, self.current_image)
            
            # Khởi tạo mô hình
            model = UNet(n_channels=3, n_classes=1).to(device)
            model.load_state_dict(torch.load('./models/unet_best-8-5-16-26.pth', map_location=device, weights_only=True))
            model.eval()
            
            # Đọc ảnh để xử lý với PIL
            image = Image.open(temp_image_path).convert("RGB")
            original_size = image.size
            image_np = np.array(image)
            
            # Tiền xử lý ảnh
            transform = A.Compose([
                A.Resize(256, 256),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
            
            transformed = transform(image=image_np)
            image_tensor = transformed["image"].unsqueeze(0).to(device)
            
            # Dự đoán mask
            with torch.no_grad():
                pred_mask_logits = model(image_tensor)
                pred_mask_probs = torch.sigmoid(pred_mask_logits)
                pred_mask = (pred_mask_probs > 0.5).float().squeeze().cpu().numpy()
            
            # Resize mask về kích thước gốc
            pred_mask_resized = Image.fromarray((pred_mask * 255).astype(np.uint8)).resize(original_size, Image.NEAREST)
            pred_mask_resized_np = np.array(pred_mask_resized)
            pred_mask_binary_resized = (pred_mask_resized_np > 127).astype(np.uint8)
            
            # Tạo overlay màu
            # overlayed_image, colored_mask = self.visualize_segmentation_map_binary(image_np, pred_mask_binary_resized)
            
            # Chuyển mask nhị phân thành ảnh 3 kênh để hiển thị đúng
            mask_3channel = cv2.cvtColor(pred_mask_binary_resized.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
            self.current_image = mask_3channel

            
            # Cập nhật ảnh hiện tại - sử dụng overlayed_image thay vì mask nhị phân
            # self.current_image = overlayed_image
            
            # Xóa file tạm
            import os
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
            
            self.display_image(self.current_image)
            self.update_ui_state()



    def apply_enhancement(self):
        self.current_step = 3  # Enhancement complete
        self.progress_bar.setValue(self.current_step)
        self.progress_bar.setFormat(self.step_names[self.current_step])

        if self.current_image is None:
            return
        
        if self.enhance_hist_btn.isChecked():
            # Apply histogram equalization
            lab = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            enhanced_lab = cv2.merge((cl, a, b))
            self.current_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        elif self.enhance_median_btn.isChecked():
            # Apply median filter
            self.current_image = cv2.medianBlur(self.current_image, 5)
        
        self.current_step = 3  # Enhancement complete
        self.progress_bar.setValue(self.current_step)
        self.display_image(self.current_image)
        self.update_ui_state()
    
    def apply_morphology(self):
        self.current_step = 4  # Morphology complete
        self.progress_bar.setValue(self.current_step)
        self.progress_bar.setFormat(self.step_names[self.current_step])

        if self.current_image is None:
            return
        
        kernel_size = self.kernel_size_slider.value()
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Convert to grayscale if not already
        if len(self.current_image.shape) == 3:
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.current_image.copy()
        
        operation = self.morph_combo.currentText()
        
        if operation == "Erosion":
            result = cv2.erode(gray, kernel, iterations=1)
        elif operation == "Dilation":
            result = cv2.dilate(gray, kernel, iterations=1)
        elif operation == "Opening":
            result = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        elif operation == "Closing":
            result = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # Convert back to BGR if original was BGR
        if len(self.current_image.shape) == 3:
            self.current_image = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        else:
            self.current_image = result
        
        self.current_step = 4  # Morphology complete
        self.progress_bar.setValue(self.current_step)
        self.display_image(self.current_image)
        self.update_ui_state()
    
    def apply_compression(self):
        self.current_step = 5  # Compression complete
        self.progress_bar.setValue(self.current_step)
        self.progress_bar.setFormat(self.step_names[self.current_step])
        if self.current_image is None:
            return
        
        quality = self.compression_slider.value()
        
        # For demonstration, we'll simulate compression by saving and reloading
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded_img = cv2.imencode('.jpg', self.current_image, encode_param)
        self.current_image = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
        
        self.current_step = 5  # Compression complete
        self.progress_bar.setValue(self.current_step)
        self.display_image(self.current_image)
        self.update_ui_state()
    
    def save_image(self):
        if self.current_image is None:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "JPEG (*.jpg);;PNG (*.png);;All Files (*)")
        
        if file_path:
            cv2.imwrite(file_path, self.current_image)
    
    def display_project_info(self):
        # Xóa pixmap hiện tại nếu có
        self.image_label.clear()
        
        # Tạo một QPixmap mới với kích thước cố định
        fixed_width = 800
        fixed_height = 600
        pixmap = QPixmap(fixed_width, fixed_height)
        pixmap.fill(Qt.white)  # Đặt nền trắng
        
        # Tạo QPainter để vẽ lên pixmap
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Đặt font và màu sắc
        title_font = painter.font()
        title_font.setPointSize(24)
        title_font.setBold(True)
        
        subtitle_font = painter.font()
        subtitle_font.setPointSize(16)
        subtitle_font.setBold(True)
        
        normal_font = painter.font()
        normal_font.setPointSize(12)
        
        # Vẽ tiêu đề dự án
        painter.setFont(title_font)
        painter.setPen(QColor("#2196F3"))
        painter.drawText(QRect(0, 150, fixed_width, 50), Qt.AlignCenter, "Medical image segmentation")
        
        # Vẽ thông tin nhóm
        painter.setFont(subtitle_font)
        painter.setPen(QColor("#333333"))
        painter.drawText(QRect(0, 210, fixed_width, 40), Qt.AlignCenter, "GROUP 5 - DIPR430685E")
        
        # Vẽ thông tin học kỳ
        painter.setFont(normal_font)
        painter.drawText(QRect(0, 260, fixed_width, 30), Qt.AlignCenter, "Semester 2 - ACADEMIC YEAR 2024-2025")
        
        # Vẽ thông tin giảng viên
        painter.drawText(QRect(0, 290, fixed_width, 30), Qt.AlignCenter, "LECTURER: HOANG VAN DUNG")
        
        # Vẽ thông tin sinh viên
        painter.drawText(QRect(0, 340, fixed_width, 30), Qt.AlignCenter, "Nguyễn Mai Huy Hoàng - 22110028")
        painter.drawText(QRect(0, 370, fixed_width, 30), Qt.AlignCenter, "Nguyễn Nhật An - 22110007")
        painter.drawText(QRect(0, 400, fixed_width, 30), Qt.AlignCenter, "Trần Trung Tín - 22110076")
        painter.drawText(QRect(0, 430, fixed_width, 30), Qt.AlignCenter, "Đinh Tô Quốc Thắng - 22110070")
        
        # Vẽ hướng dẫn
        painter.setPen(QColor("#666666"))
        painter.drawText(QRect(0, 480, fixed_width, 30), Qt.AlignCenter, "Click 'Open Image' to start processing")
        
        # Kết thúc vẽ
        painter.end()
        
        # Hiển thị pixmap
        self.image_label.setPixmap(pixmap)
        self.image_label.setAlignment(Qt.AlignCenter)
        
        # Loại bỏ đường viền đứt nét
        self.image_label.setStyleSheet("border: none; font-size: 18px;")

    def display_image(self, image):
        if image is None:
            return
        
        # Convert the image to RGB (from BGR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create QImage from the image data
        h, w, c = image_rgb.shape
        bytes_per_line = c * w
        q_image = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Create QPixmap from QImage
        pixmap = QPixmap.fromImage(q_image)
        
        # Sử dụng kích thước cố định
        fixed_width = 800
        fixed_height = 600
        
        # Scale pixmap với kích thước cố định
        pixmap = pixmap.scaled(fixed_width, fixed_height, 
                            Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # Loại bỏ đường viền đứt nét
        self.image_label.setStyleSheet("border: none; font-size: 18px;")
        
        # Hiển thị pixmap
        self.image_label.setPixmap(pixmap)
        self.image_label.setAlignment(Qt.AlignCenter)

    def resizeEvent(self, event):
        # Update image display when window is resized
        if hasattr(self, 'current_image') and self.current_image is not None:
            self.display_image(self.current_image)
        super().resizeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = BloodCellSegmentationApp()
    window.show()
    sys.exit(app.exec_())
