import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from ultralytics import YOLO

def create_small_dataset():
    """Create a smaller dataset for faster training."""
    print("\n=== Creating smaller dataset for faster training ===")
    
    # Define directories
    train_dir = Path("train")
    val_dir = Path("val")
    small_train_dir = Path("small_train")
    small_val_dir = Path("small_val")
    
    # Create directories if they don't exist
    os.makedirs(small_train_dir, exist_ok=True)
    os.makedirs(small_val_dir, exist_ok=True)
    
    # Get all image files
    train_images = list(train_dir.glob("*.png"))
    val_images = list(val_dir.glob("*.png"))
    
    # Use only 10% of the dataset
    train_sample = train_images[:min(600, len(train_images))]
    val_sample = val_images[:min(150, len(val_images))]
    
    print(f"Using {len(train_sample)} training images and {len(val_sample)} validation images")
    
    # Copy files to smaller dataset directories
    for img_path in train_sample:
        img_filename = img_path.name
        txt_filename = img_path.stem + ".txt"
        txt_path = train_dir / txt_filename
        
        # Copy image and annotation to small train directory
        os.system(f"cp {img_path} {small_train_dir / img_filename}")
        os.system(f"cp {txt_path} {small_train_dir / txt_filename}")
    
    for img_path in val_sample:
        img_filename = img_path.name
        txt_filename = img_path.stem + ".txt"
        txt_path = val_dir / txt_filename
        
        # Copy image and annotation to small validation directory
        os.system(f"cp {img_path} {small_val_dir / img_filename}")
        os.system(f"cp {txt_path} {small_val_dir / txt_filename}")
    
    print("Small dataset created successfully!")
    
    # Create YAML config file for the small dataset
    create_yaml_config()

def create_yaml_config():
    """Create YAML configuration file for the small dataset."""
    yaml_content = f"""path: {os.getcwd()}
train: small_train
val: small_val

names:
  0: Car
  1: Van
  2: Truck
  3: Pedestrian
  4: Person_sitting
  5: Cyclist
  6: Tram
  7: Misc
"""
    
    with open("small_kitti.yaml", "w") as f:
        f.write(yaml_content)
    
    print("Created small_kitti.yaml configuration file")

def train_yolov8():
    """Train YOLOv8 model with optimized settings for faster training."""
    print("\n=== Training YOLOv8 (Fast Version) ===")
    model = YOLO('models/yolov8n.pt')  # Using the smallest YOLOv8 model
    results = model.train(
        data='small_kitti.yaml',
        epochs=10,
        imgsz=320,  # Smaller image size
        batch=8,    # Smaller batch size
        name='yolov8_kitti_fast',
        device='cpu',
        optimizer='auto',
        verbose=True,
        seed=0,
        deterministic=True,
        single_cls=False,
        rect=False,
        cos_lr=False,
        close_mosaic=10,
        resume=False,
        amp=True,
        fraction=1.0,
        project='runs/train'
    )
    return results

def evaluate_model(model_path, yaml_path):
    """Evaluate a trained model and generate metrics."""
    print(f"\n=== Evaluating YOLOv8 ===")
    
    # Define classes
    classes = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']
    
    # Load the trained model
    model = YOLO(model_path)
    
    # Validate model
    print(f"Validating YOLOv8...")
    valid_results = model.val(data=yaml_path)
    
    # Extract metrics
    print(f"Extracting metrics for YOLOv8...")
    metrics = {}
    metrics['precision'] = valid_results.box.p
    metrics['recall'] = valid_results.box.r
    metrics['f1_score'] = valid_results.box.f1
    metrics['ap50'] = valid_results.box.ap50
    metrics['ap'] = valid_results.box.ap
    metrics['mp'] = valid_results.box.mp
    metrics['mr'] = valid_results.box.mr
    metrics['map50'] = valid_results.box.map50
    metrics['map'] = valid_results.box.map
    
    # Process metrics for each class
    print(f"Processing metrics for each class in YOLOv8...")
    class_metrics = []
    for i, class_name in enumerate(classes):
        class_metrics.append({
            'class': class_name,
            'precision': metrics['precision'][i],
            'recall': metrics['recall'][i],
            'f1_score': metrics['f1_score'][i],
            'ap50': metrics['ap50'][i],
            'ap': metrics['ap'][i]
        })
    
    # Add overall metrics
    class_metrics.append({
        'class': 'Overall',
        'precision': metrics['mp'],
        'recall': metrics['mr'],
        'f1_score': 2 * (metrics['mp'] * metrics['mr']) / (metrics['mp'] + metrics['mr'] + 1e-6),
        'ap50': metrics['map50'],
        'ap': metrics['map']
    })
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(class_metrics)
    metrics_file = f"yolov8_metrics_fast.csv"
    metrics_df.to_csv(metrics_file, index=False)
    
    # Save confusion matrix if available
    cm_file = f"yolov8_confusion_matrix_fast.png"
    if hasattr(valid_results, 'confusion_matrix'):
        print(f"Generating confusion matrix for YOLOv8...")
        cm = valid_results.confusion_matrix.matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', 
                    xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('YOLOv8 Confusion Matrix')
        plt.tight_layout()
        plt.savefig(cm_file)
        plt.close()
    
    print(f"\nYOLOv8 evaluation completed!")
    print(f"Metrics saved to: {metrics_file}")
    if hasattr(valid_results, 'confusion_matrix'):
        print(f"Confusion matrix saved to: {cm_file}")
    
    return metrics_df

