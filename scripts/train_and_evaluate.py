import os
import shutil
import subprocess
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from ultralytics import YOLO

def extract_files_from_backup():
    """Extract necessary files from the backup folder."""
    print("\n=== Extracting files from backup folder ===")
    backup_dir = Path("backup")
    
    # Files to extract
    files_to_extract = [
        "train_all_yolo.py",
        "yolov3_metrics.py",
        "yolov5_metrics.py",
        "yolov8_metrics.py"
    ]
    
    for file in files_to_extract:
        source = backup_dir / file
        if source.exists():
            shutil.copy(source, Path(file))
            print(f"Extracted: {file}")
        else:
            print(f"Warning: {file} not found in backup folder")
    
    print("File extraction completed")

def train_yolov3():
    """Train YOLOv3 model."""
    print("\n=== Training YOLOv3 ===")
    model = YOLO('models/yolov3.pt')
    results = model.train(
        data='kitti_v3.yaml',
        epochs=10,
        imgsz=416,
        batch=16,
        name='yolov3_kitti',
        device='cpu',
        optimizer='AdamW',
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

def train_yolov5():
    """Train YOLOv5 model."""
    print("\n=== Training YOLOv5 ===")
    model = YOLO('models/yolov5nu.pt')
    results = model.train(
        data='kitti_v5.yaml',
        epochs=10,
        imgsz=640,
        batch=16,
        name='yolov5_kitti',
        device='cpu',
        optimizer='AdamW',
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

def train_yolov8():
    """Train YOLOv8 model."""
    print("\n=== Training YOLOv8 ===")
    model = YOLO('models/yolov8n.pt')
    results = model.train(
        data='kitti.yaml',
        epochs=10,
        imgsz=640,
        batch=16,
        name='yolov8_kitti',
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

def evaluate_model(model_name, model_path, yaml_path):
    """Evaluate a trained model and generate metrics."""
    print(f"\n=== Evaluating {model_name} ===")
    
    # Define classes
    classes = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']
    
    # Load the trained model
    model = YOLO(model_path)
    
    # Validate model
    print(f"Validating {model_name}...")
    valid_results = model.val(data=yaml_path)
    
    # Extract metrics
    print(f"Extracting metrics for {model_name}...")
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
    print(f"Processing metrics for each class in {model_name}...")
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
    metrics_file = f"{model_name.lower()}_metrics.csv"
    metrics_df.to_csv(metrics_file, index=False)
    
    # Save confusion matrix if available
    cm_file = f"{model_name.lower()}_confusion_matrix.png"
    if hasattr(valid_results, 'confusion_matrix'):
        print(f"Generating confusion matrix for {model_name}...")
        cm = valid_results.confusion_matrix.matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', 
                    xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'{model_name} Confusion Matrix')
        plt.tight_layout()
        plt.savefig(cm_file)
        plt.close()
    
    print(f"\n{model_name} evaluation completed!")
    print(f"Metrics saved to: {metrics_file}")
    if hasattr(valid_results, 'confusion_matrix'):
        print(f"Confusion matrix saved to: {cm_file}")
    
    return metrics_df, valid_results

def generate_comparison_plots(metrics_dfs, model_names):
    """Generate comparison plots for all models."""
    print("\n=== Generating Comparison Plots ===")
    
    # Extract overall metrics for each model
    overall_metrics = []
    for i, df in enumerate(metrics_dfs):
        overall = df[df['class'] == 'Overall'].iloc[0].to_dict()
        overall['model'] = model_names[i]
        overall_metrics.append(overall)
    
    comparison_df = pd.DataFrame(overall_metrics)
    
    # Create comparison bar plots
    metrics_to_plot = ['precision', 'recall', 'f1_score', 'ap50', 'ap']
    labels = ['Precision', 'Recall', 'F1-Score', 'AP@50', 'mAP']
    
    # Bar plot for all metrics
    plt.figure(figsize=(12, 8))
    x = np.arange(len(metrics_to_plot))
    width = 0.2
    
    for i, model in enumerate(model_names):
        values = [comparison_df[comparison_df['model'] == model][metric].values[0] for metric in metrics_to_plot]
        plt.bar(x + i*width, values, width, label=model)
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x + width, labels)
    plt.legend()
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()
    
    # Individual metric plots
    for i, metric in enumerate(metrics_to_plot):
        plt.figure(figsize=(10, 6))
        values = [comparison_df[comparison_df['model'] == model][metric].values[0] for model in model_names]
        plt.bar(model_names, values, color=['blue', 'green', 'red'])
        plt.xlabel('Model')
        plt.ylabel(labels[i])
        plt.title(f'{labels[i]} Comparison')
        plt.tight_layout()
        plt.savefig(f'{metric}_comparison.png')
        plt.close()
    
    # Save comparison to CSV
    comparison_df.to_csv('model_comparison.csv', index=False)
    
    print("Comparison plots generated:")
    print("- model_comparison.png")
    for metric in metrics_to_plot:
        print(f"- {metric}_comparison.png")
    print("Comparison data saved to: model_comparison.csv")

def create_readme():
    """Create a README file with information about the project and results."""
    print("\n=== Creating README file ===")
    
    readme_content = """# SadakSaathi YOLOv3, YOLOv5, and YOLOv8 Model Comparison

## Project Overview
This project trains and evaluates YOLOv3, YOLOv5, and YOLOv8 models on the KITTI dataset for object detection. The models are trained to detect various objects including cars, pedestrians, cyclists, and more.

## Dataset
The dataset used is the KITTI dataset, which contains images of road scenes with annotations for various objects. The dataset is organized in YOLO format with image files (.png) and corresponding annotation files (.txt).

## Models
Three different YOLO models were trained and evaluated:
- YOLOv3: A classic object detection model
- YOLOv5: An improved version with better performance
- YOLOv8: The latest version with state-of-the-art performance

## Training
Each model was trained for 10 epochs with the following parameters:
- YOLOv3: Image size 416x416, AdamW optimizer
- YOLOv5: Image size 640x640, AdamW optimizer
- YOLOv8: Image size 640x640, Auto optimizer

## Evaluation Metrics
The models were evaluated using the following metrics:
- Precision: The ability of the model to identify only relevant objects
- Recall: The ability of the model to find all relevant objects
- F1-Score: The harmonic mean of precision and recall
- AP@50: Average Precision at IoU threshold of 0.5
- mAP: Mean Average Precision across IoU thresholds

## Results
The performance comparison between the models can be found in the following files:
- `model_comparison.png`: Overall comparison of all metrics
- `precision_comparison.png`: Comparison of precision scores
- `recall_comparison.png`: Comparison of recall scores
- `f1_score_comparison.png`: Comparison of F1 scores
- `ap50_comparison.png`: Comparison of AP@50 scores
- `ap_comparison.png`: Comparison of mAP scores

Detailed metrics for each model can be found in:
- `yolov3_metrics.csv`
- `yolov5_metrics.csv`
- `yolov8_metrics.csv`

Confusion matrices for each model:
- `yolov3_confusion_matrix.png`
- `yolov5_confusion_matrix.png`
- `yolov8_confusion_matrix.png`

## Conclusion
Based on the evaluation metrics, the best performing model is determined by comparing the overall precision, recall, F1-score, AP@50, and mAP values. The model with the highest scores across these metrics is considered the most effective for the given task.

## How to Run
To train and evaluate the models, run:
```
python train_and_evaluate.py
```

This script will:
1. Extract necessary files from the backup folder
2. Train YOLOv3, YOLOv5, and YOLOv8 models
3. Evaluate each model and generate metrics
4. Create comparison plots
5. Generate this README file
"""
    
    with open('README.md', 'w') as f:
        f.write(readme_content)
    
    print("README.md file created")

def main():
    """Main function to run the entire process."""
    start_time = time.time()
    
    print("=== SadakSaathi YOLOv3, YOLOv5, and YOLOv8 Training and Evaluation ===")
    
    # Step 1: Extract files from backup
    extract_files_from_backup()
    
    # Step 2: Train models
    print("\n=== Training Models ===")
    try:
        # Train YOLOv3
        yolov3_results = train_yolov3()
        print("YOLOv3 training completed!")
        
        # Train YOLOv5
        yolov5_results = train_yolov5()
        print("YOLOv5 training completed!")
        
        # Train YOLOv8
        yolov8_results = train_yolov8()
        print("YOLOv8 training completed!")
        
        print("\nAll models have been trained successfully!")
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        return
    
    # Step 3: Evaluate models and generate metrics
    print("\n=== Evaluating Models ===")
    try:
        # Evaluate YOLOv3
        yolov3_metrics, yolov3_val_results = evaluate_model(
            "YOLOv3", 
            "runs/train/yolov3_kitti/weights/best.pt", 
            "kitti_v3.yaml"
        )
        
        # Evaluate YOLOv5
        yolov5_metrics, yolov5_val_results = evaluate_model(
            "YOLOv5", 
            "runs/train/yolov5_kitti/weights/best.pt", 
            "kitti_v5.yaml"
        )
        
        # Evaluate YOLOv8
        yolov8_metrics, yolov8_val_results = evaluate_model(
            "YOLOv8", 
            "runs/train/yolov8_kitti/weights/best.pt", 
            "kitti.yaml"
        )
    except Exception as e:
        print(f"An error occurred during evaluation: {str(e)}")
        return
    
    # Step 4: Generate comparison plots
    try:
        generate_comparison_plots(
            [yolov3_metrics, yolov5_metrics, yolov8_metrics],
            ["YOLOv3", "YOLOv5", "YOLOv8"]
        )
    except Exception as e:
        print(f"An error occurred during plot generation: {str(e)}")
        return
    
    # Step 5: Create README
    create_readme()
    
    # Calculate and display total execution time
    end_time = time.time()
    execution_time = end_time - start_time
    hours, remainder = divmod(execution_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n=== Process Completed ===")
    print(f"Total execution time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print("\nCheck the README.md file for information about the results and how to interpret them.")

if __name__ == "__main__":
    main()
