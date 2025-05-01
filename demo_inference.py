# demo_inference.py
"""
Demo script for running inference on images or video
"""

from ultralytics import YOLO

def run_inference(model_path, source_path):
    """Runs inference on a folder of images or video."""
    model = YOLO(model_path)

    results = model.predict(
        source=source_path, 
        save=True, 
        imgsz=416, 
        conf=0.25
    )

    print("[INFO] Inference complete. Outputs saved.")

if __name__ == "__main__":
    model_path = "helmet_detection_project/yolov8n_helmet_detection/weights/best.onnx"
    source_path = "C:/Users/nanth/yolo/3-obreros-reunidos-en-su-centro-de-trabajo_jpg.rf.d39598f69efbd7fd7a079e1e4b59f0a0.jpg"  # or a video file path
    run_inference(model_path, source_path)
