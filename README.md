# SadakSaathi - Road Safety AI assistant

## Project Overview
SadakSaathi is a comprehensive object detection system designed for road safety applications. The project uses state-of-the-art YOLO (You Only Look Once) models to detect and classify various road objects such as cars, pedestrians, cyclists, and more from the KITTI dataset.

## Dataset
link for the dataset : https://www.kaggle.com/datasets/klemenko/kitti-dataset
,The project uses the KITTI dataset, which contains images of road scenes with annotations for various objects:
- Total training images: 5,984
- Total validation images: 1,497

The dataset includes the following classes:
- Car
- Van
- Truck
- Pedestrian
- Person_sitting
- Cyclist
- Tram
- Misc

## Models
The project implements and compares three different YOLO models:

### YOLOv3
- Architecture: Classic object detection model
- Image size: 416x416
- Optimizer: AdamW

### YOLOv5
- Architecture: Improved version with better performance
- Image size: 640x640
- Optimizer: AdamW

### YOLOv8
- Architecture: Latest version with state-of-the-art performance
- Image size: 640x640
- Optimizer: Auto

## Project Structure
```
SadakSaathi/
├── backup/                  # Backup of original scripts
├── models/                  # Pre-trained model weights
├── runs/                    # Training and validation results
├── train/                   # Training dataset
├── val/                     # Validation dataset
├── kitti.yaml               # Dataset configuration for YOLOv8
├── kitti_v3.yaml            # Dataset configuration for YOLOv3
├── kitti_v5.yaml            # Dataset configuration for YOLOv5
├── train_and_evaluate.py    # Main script for training and evaluation
├── fast_train_yolov8.py     # Fast training script for YOLOv8
└── split_dataset.py         # Script to split dataset
```

## Training
Each model is trained for 10 epochs with the following common parameters:
- Batch size: 16
- Deterministic: True
- Seed: 0
- Single class: False

## Evaluation Metrics
The models are evaluated using the following metrics:
- Precision: The ability of the model to identify only relevant objects
- Recall: The ability of the model to find all relevant objects
- F1-Score: The harmonic mean of precision and recall
- AP@50: Average Precision at IoU threshold of 0.5
- mAP: Mean Average Precision across IoU thresholds

## Results
The performance comparison between the models can be found in:
- `model_comparison.png`: Overall comparison of all metrics
- Individual metric comparison plots for precision, recall, F1-score, AP@50, and mAP

Detailed metrics for each model:
- `yolov3_metrics.csv`
- `yolov5_metrics.csv`
- `yolov8_metrics.csv`

Confusion matrices:
- `yolov3_confusion_matrix.png`
- `yolov5_confusion_matrix.png`
- `yolov8_confusion_matrix.png`

### Sample Training Results

#### YOLOv3 Results
![YOLOv3 Prediction](/Users/simarpreetsingh/Desktop/Sadaksaathi/runs/train/yolov3_kitti22/val_batch0_pred.jpg)
*YOLOv3 object detection predictions*

![YOLOv3 Confusion Matrix](/Users/simarpreetsingh/Desktop/Sadaksaathi/runs/train/yolov3_kitti22/confusion_matrix_normalized.png)
*YOLOv3 confusion matrix showing classification performance*

#### YOLOv8 Results
![YOLOv8 Prediction](/Users/simarpreetsingh/Desktop/Sadaksaathi/runs/train/yolov8_kitti92/val_batch0_pred.jpg)
*YOLOv8 object detection predictions*

![YOLOv8 Precision-Recall Curve](/Users/simarpreetsingh/Desktop/Sadaksaathi/runs/train/yolov8_kitti92/PR_curve.png)
*YOLOv8 precision-recall curve showing model performance*

## How to Run

### Full Training (All Models)
This will train all three models (YOLOv3, YOLOv5, YOLOv8) and generate comparison metrics:
```bash
python3 train_and_evaluate.py
```
Note: Full training takes approximately 3 days on CPU.

### Fast Training (YOLOv8 Only)
For a quicker result (completes in under 2 hours):
```bash
python3 fast_train_yolov8.py
```

### Dataset Preparation
If you need to split the dataset manually:
```bash
python3 split_dataset.py
```

## Requirements
- Python 3.x
- PyTorch
- Ultralytics
- NumPy
- Pandas
- Matplotlib
- Seaborn

## Future Improvements
- Implement real-time detection using webcam
- Add support for more object classes
- Optimize models for edge devices
- Integrate with traffic management systems

## License
This project is licensed under the MIT License - see the LICENSE file for details.
