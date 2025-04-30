# evaluate.py
"""
Evaluation script for Helmet Detection model
"""

from ultralytics import YOLO

def evaluate_model(model_path, data_yaml_path):
    """Evaluates the model and prints metrics."""
    model = YOLO(model_path)

    # Evaluate on test set
    metrics = model.val(
        data=data_yaml_path,
        split='test',  # Use the test split
        plots=True,    # Save confusion matrix, PR curves
        save_json=True # Save COCO-style JSON output
    )

    print("[INFO] Evaluation Metrics:")
    print(metrics)

if __name__ == "__main__":
    model_path = "helmet_detection_project/yolov8n_helmet_detection/weights/best.pt"  # <-- Update
    data_yaml_path = "C:/Users/nanth/yolo/dataset/data.yaml"
    evaluate_model(model_path, data_yaml_path)
