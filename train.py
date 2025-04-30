# train.py
"""
Training script for Helmet Detection using YOLOv8
Author: [Your Name]
"""

import os
import zipfile
from ultralytics import YOLO

def train_model(data_yaml_path, model_size='n'):
    """Trains YOLOv8 model with optimization strategies."""
    model = YOLO(f'yolov8{model_size}.pt')  # model_size: 'n', 's', 'm', 'l', 'x'

    model.train(
        data=data_yaml_path,
        epochs=20,
        imgsz=416,
        batch=8,
        patience=10,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        device="cpu",  # 'cpu' if no GPU
        workers=4,
        project="helmet_detection_project",
        name=f"yolov8{model_size}_helmet_detection",
        pretrained=True,
        val=True,
        save_period=10,
        verbose=True,
        exist_ok=True,
        cos_lr=True,
        close_mosaic=2,
        amp=True
    )

if __name__ == "__main__":
    extract_path = "C:/Users/nanth/yolo/dataset/data.yaml"

    train_model(extract_path, model_size='n')
