import os
import sys
import argparse
from pathlib import Path
from ultralytics import YOLO

def predict_image(model_path, image_path, conf_threshold=0.25):
    """
    Run prediction on a single image using a trained YOLO model.
    
    Args:
        model_path: Path to the trained model weights
        image_path: Path to the image for prediction
        conf_threshold: Confidence threshold for detections
    """
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    # Load the model
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    # Run prediction
    print(f"Running prediction on {image_path}...")
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        save=True,
        save_txt=True,
        save_conf=True
    )
    
    # Print results
    print("\nPrediction Results:")
    for r in results:
        boxes = r.boxes
        print(f"Detected {len(boxes)} objects")
        
        # Print each detection
        for box in boxes:
            cls = int(box.cls[0])
            cls_name = model.names[cls]
            conf = float(box.conf[0])
            coords = box.xyxy[0].tolist()
            print(f"- {cls_name}: {conf:.2f} at {[round(c, 2) for c in coords]}")
    
    # Print output path
    output_path = Path(model.predictor.save_dir)
    print(f"\nResults saved to: {output_path}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run prediction using trained YOLO model")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model weights (.pt file)")
    parser.add_argument("--image", type=str, required=True, help="Path to image for prediction")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    args = parser.parse_args()
    
    # Run prediction
    predict_image(args.model, args.image, args.conf)

if __name__ == "__main__":
    main()
