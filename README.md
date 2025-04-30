# 🪖 construction object Detection with YOLOv8

This project implements a **construction object detection ** using the yolov8 object detection framework. It covers end-to-end steps from training and evaluation to ONNX deployment and live inference.

---

## 📂 Project Structure

├── dataset/ # Annotated dataset (YOLO format) ├── runs/ # Training logs and best.pt weights ├── train.py # YOLOv8 training script 
├── demo_inference.py # Inference script for images or video 
├── deploy.py # Model export script to ONNX 
├── evaluation.py # Evaluation report ├── results/ # Demo inference images ├── weights/ │ ├── best.pt # Trained PyTorch model │
|── best.onnx # ONNX exported model 

1.pip install -r requirements.txt
2.run train.py # alter the dataset path and model selection
3.run deploy.py # to covert onnx  format 
4. run demo_inference.py # add image or video 
5. evaluate.py # get result for training 

![3-obreros-reunidos-en-su-centro-de-trabajo_jpg rf d39598f69efbd7fd7a079e1e4b59f0a0](https://github.com/user-attachments/assets/68cb9d78-db96-4b73-bc27-50f33e0c1af3)
![results](https://github.com/user-attachments/assets/a7cceebd-01e2-4b97-91a3-c3d8d587dfa7)

