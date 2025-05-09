import os
import random
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import torch
from PIL import Image
from evaluate import visualize_prediction_binary_with_accuracy, UNet
import albumentations as A
from albumentations.pytorch import ToTensorV2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# filepath: d:\tailieu3\HK2-24-25\Digital Image Processing\FinalProject\Test_U-Net-version-1\test_evaluate.py
import matplotlib.pyplot as plt

# Import function from evaluate.py (using absolute import)

def test_model_on_random_samples(model_path, dataset_dir, num_samples=100, save_results=True):
    """
    Test the U-Net model on a random subset of images from the dataset
    and visualize performance metrics.
    
    Args:
        model_path: Path to the trained model
        dataset_dir: Directory containing image and mask subdirectories
        num_samples: Number of random samples to test (default: 100)
        save_results: Whether to save result plots (default: True)
    """
    # Set up device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔹 Using device: {device}")
    
    # Load model
    model = UNet(n_channels=3, n_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Get list of all images
    image_dir = os.path.join(dataset_dir, "image")
    mask_dir = os.path.join(dataset_dir, "mask")
    
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    print(f"✅ Found {len(image_files)} images in the dataset")
    
    # Select random samples
    if num_samples > len(image_files):
        num_samples = len(image_files)
        print(f"⚠️ Requested more samples than available. Using all {num_samples} images.")
    
    random_samples = random.sample(image_files, num_samples)
    print(f"✅ Selected {len(random_samples)} random images for testing")
    
    # Initialize arrays to store metrics
    results = {
        'image_name': [],
        'accuracy': [],
        'iou': [],
        'dice': [],
        'precision': [],
        'recall': []
    }
    
    # Process each image and collect metrics
    for image_file in tqdm(random_samples, desc="Processing images"):
        image_path = os.path.join(image_dir, image_file)
        mask_file = image_file.replace('image', 'mask')
        mask_path = os.path.join(mask_dir, mask_file)
        
        # Skip if mask doesn't exist
        if not os.path.exists(mask_path):
            print(f"⚠️ Mask not found for {image_file}, skipping...")
            continue
        
        # Process the image but suppress visualization
        with plt.ioff():  # Turn off interactive plotting
            metrics = evaluate_single_image(model, image_path, mask_path, device)
        
        # Store results
        results['image_name'].append(image_file)
        results['accuracy'].append(metrics['accuracy'])
        results['iou'].append(metrics['iou'])
        results['dice'].append(metrics['dice']) 
        results['precision'].append(metrics['precision'])
        results['recall'].append(metrics['recall'])
    
    # Convert results to DataFrame for easier analysis
    df = pd.DataFrame(results)
    
    # Create directory for results
    if save_results:
        os.makedirs("./test_results", exist_ok=True)
        df.to_csv("./test_results/metrics_summary.csv", index=False)
    
    # Visualize metrics
    visualize_metrics(df, save_results)
    
    # Show best and worst examples
    show_best_worst_samples(df, image_dir, mask_dir, model, device, save_results)
    
    return df

def evaluate_single_image(model, image_path, mask_path, device):
    """Calculate metrics for a single image without displaying plots"""
    # Load image & mask
    image = Image.open(image_path).convert("RGB")
    mask_true = Image.open(mask_path).convert("L")
    original_size = image.size
    
    mask_true_np = np.array(mask_true)
    mask_true_binary = (mask_true_np > 0).astype(np.uint8)
    
    # Preprocessing & prediction code from visualize_prediction_binary_with_accuracy
    # but without visualization parts
    
    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    image_np = np.array(image)
    transformed = transform(image=image_np)
    image_tensor = transformed["image"].unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred_mask_logits = model(image_tensor)
        pred_mask_probs = torch.sigmoid(pred_mask_logits)
        pred_mask = (pred_mask_probs > 0.5).float().squeeze().cpu().numpy()
    
    # Resize mask to original size
    pred_mask_resized = Image.fromarray((pred_mask * 255).astype(np.uint8)).resize(original_size, Image.NEAREST)
    pred_mask_resized_np = np.array(pred_mask_resized)
    pred_mask_binary_resized = (pred_mask_resized_np > 127).astype(np.uint8)
    
    # Calculate metrics
    correct_pixels = np.sum(pred_mask_binary_resized == mask_true_binary)
    total_pixels = mask_true_binary.size
    accuracy = correct_pixels / total_pixels
    
    intersection = np.sum(pred_mask_binary_resized * mask_true_binary)
    union = np.sum(pred_mask_binary_resized) + np.sum(mask_true_binary) - intersection
    
    iou = intersection / union if union > 0 else 0.0
    dice = (2 * intersection) / (np.sum(pred_mask_binary_resized) + np.sum(mask_true_binary)) if (np.sum(pred_mask_binary_resized) + np.sum(mask_true_binary)) > 0 else 0.0
    
    tp = intersection
    fp = np.sum(pred_mask_binary_resized) - tp
    fn = np.sum(mask_true_binary) - tp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'iou': iou,
        'dice': dice,
        'precision': precision,
        'recall': recall
    }

def visualize_metrics(df, save_results=True):
    """Create visualizations for the collected metrics"""
    plt.figure(figsize=(15, 10))
    
    # 1. Box plots for each metric
    plt.subplot(2, 2, 1)
    sns.boxplot(data=df[['accuracy', 'iou', 'dice', 'precision', 'recall']])
    plt.title("Distribution of Metrics")
    plt.ylabel("Score")
    plt.grid(alpha=0.3)
    
    # 2. Histogram of IoU scores
    plt.subplot(2, 2, 2)
    sns.histplot(df['iou'], bins=20, kde=True)
    plt.title("IoU Distribution")
    plt.xlabel("IoU Score")
    plt.grid(alpha=0.3)
    
    # 3. Scatter plot of precision vs recall
    plt.subplot(2, 2, 3)
    sns.scatterplot(data=df, x='precision', y='recall', hue='iou', palette='viridis')
    plt.title("Precision vs. Recall")
    plt.grid(alpha=0.3)
    
    # 4. Correlation heatmap
    plt.subplot(2, 2, 4)
    correlation_matrix = df[['accuracy', 'iou', 'dice', 'precision', 'recall']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Between Metrics")
    
    plt.tight_layout()
    if save_results:
        plt.savefig("./test_results/metrics_visualization.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Additional visualization: Performance distribution by metric
    plt.figure(figsize=(12, 8))
    metrics = ['accuracy', 'iou', 'dice', 'precision', 'recall']
    
    for i, metric in enumerate(metrics):
        plt.subplot(2, 3, i+1)
        sns.violinplot(y=df[metric])
        plt.title(f"{metric.capitalize()} Distribution")
        plt.grid(alpha=0.3)
    
    plt.tight_layout()
    if save_results:
        plt.savefig("./test_results/metrics_distributions.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate and print summary statistics
    summary_stats = df[['accuracy', 'iou', 'dice', 'precision', 'recall']].describe()
    print("\n📊 SUMMARY STATISTICS:")
    print(summary_stats)
    
    if save_results:
        summary_stats.to_csv("./test_results/summary_statistics.csv")

def show_best_worst_samples(df, image_dir, mask_dir, model, device, save_results=True):
    """Show examples of best and worst performing images"""
    # Get indices of best and worst samples by IoU
    best_indices = df.nlargest(3, 'iou').index
    worst_indices = df.nsmallest(3, 'iou').index
    
    for category, indices in [("Best", best_indices), ("Worst", worst_indices)]:
        plt.figure(figsize=(15, 10))
        plt.suptitle(f"{category} Performing Images (by IoU)", fontsize=16)
        
        for i, idx in enumerate(indices):
            image_file = df.loc[idx, 'image_name']
            image_path = os.path.join(image_dir, image_file)
            mask_file = image_file.replace('image', 'mask')
            mask_path = os.path.join(mask_dir, mask_file)
            
            # Display the original image, ground truth, and prediction
            image = Image.open(image_path).convert("RGB")
            mask_true = Image.open(mask_path).convert("L")
            
            # Get metrics for the image
            metrics = df.loc[idx]
            
            # Process image to get prediction
            pred_mask = get_prediction(model, image, device)
            
            # Display results
            plt.subplot(3, 3, i*3 + 1)
            plt.imshow(image)
            plt.title(f"Original: {image_file}")
            plt.axis("off")
            
            plt.subplot(3, 3, i*3 + 2)
            plt.imshow(np.array(mask_true) > 0, cmap='gray')
            plt.title("Ground Truth")
            plt.axis("off")
            
            plt.subplot(3, 3, i*3 + 3)
            plt.imshow(pred_mask, cmap='gray')
            plt.title(f"Prediction\nIoU: {metrics['iou']:.4f}, Dice: {metrics['dice']:.4f}")
            plt.axis("off")
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        if save_results:
            plt.savefig(f"./test_results/{category.lower()}_examples.png", dpi=300, bbox_inches='tight')
        plt.show()

def get_prediction(model, image, device):
    """Get prediction mask for an image"""
    
    original_size = image.size
    
    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    image_np = np.array(image)
    transformed = transform(image=image_np)
    image_tensor = transformed["image"].unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred_mask_logits = model(image_tensor)
        pred_mask_probs = torch.sigmoid(pred_mask_logits)
        pred_mask = (pred_mask_probs > 0.5).float().squeeze().cpu().numpy()
    
    # Resize mask to original size
    pred_mask_resized = Image.fromarray((pred_mask * 255).astype(np.uint8)).resize(original_size, Image.NEAREST)
    pred_mask_resized_np = np.array(pred_mask_resized)
    pred_mask_binary_resized = (pred_mask_resized_np > 127).astype(np.uint8)
    
    return pred_mask_binary_resized

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Test the model on 100 random images
    dataset_dir = './data/KRD-WBC dataset/Dataset'
    model_path = './models/unet_best-8-5-16-26.pth'
    
    results_df = test_model_on_random_samples(
        model_path=model_path,
        dataset_dir=dataset_dir,
        num_samples=100,  # Test on 100 random images
        save_results=True
    )
    
    # Print overall statistics
    print("\n📈 OVERALL METRICS:")
    for metric in ['accuracy', 'iou', 'dice', 'precision', 'recall']:
        print(f"Average {metric.capitalize()}: {results_df[metric].mean():.4f} ± {results_df[metric].std():.4f}")