from ultralytics import YOLO

# Load your trained model (change path if needed)
model = YOLO('helmet_detection_project/yolov8n_helmet_detection/weights/best.pt')

# Export to ONNX
model.export(format='onnx')
