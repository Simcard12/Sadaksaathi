import os
import shutil
import random
from pathlib import Path
import glob

def split_dataset(source_dir, train_dir, val_dir, split_ratio=0.8):
    """
    Split the dataset into training and validation sets.
    
    Args:
        source_dir: Source directory containing the dataset
        train_dir: Directory to store training data
        val_dir: Directory to store validation data
        split_ratio: Ratio of training data (default: 0.8)
    """
    # Create directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Get all image files
    image_files = sorted(glob.glob(os.path.join(source_dir, "*.png")))
    
    # Shuffle the files with a fixed seed for reproducibility
    random.seed(42)
    random.shuffle(image_files)
    
    # Calculate split point
    split_idx = int(len(image_files) * split_ratio)
    
    # Split into training and validation sets
    train_images = image_files[:split_idx]
    val_images = image_files[split_idx:]
    
    print(f"Total images: {len(image_files)}")
    print(f"Training images: {len(train_images)}")
    print(f"Validation images: {len(val_images)}")
    
    # Copy files to respective directories
    for img_path in train_images:
        img_filename = os.path.basename(img_path)
        txt_filename = os.path.splitext(img_filename)[0] + ".txt"
        txt_path = os.path.join(source_dir, txt_filename)
        
        # Copy image and annotation to train directory
        shutil.copy(img_path, os.path.join(train_dir, img_filename))
        shutil.copy(txt_path, os.path.join(train_dir, txt_filename))
        
    for img_path in val_images:
        img_filename = os.path.basename(img_path)
        txt_filename = os.path.splitext(img_filename)[0] + ".txt"
        txt_path = os.path.join(source_dir, txt_filename)
        
        # Copy image and annotation to validation directory
        shutil.copy(img_path, os.path.join(val_dir, img_filename))
        shutil.copy(txt_path, os.path.join(val_dir, txt_filename))
    
    print("Dataset split completed successfully!")

if __name__ == "__main__":
    # Define directories
    source_dir = "yolo_dataset"
    train_dir = "train"
    val_dir = "val"
    
    # Split the dataset
    split_dataset(source_dir, train_dir, val_dir)
