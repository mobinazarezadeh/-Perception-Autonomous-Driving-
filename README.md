# Autonomous Driving Perception Pipeline 🚗

A real-time perception system for autonomous driving built on the KITTI dataset.

## 🎯 What This Project Does
Detects and localizes vehicles, pedestrians, and cyclists in driving scenes using YOLOv8.

## 📊 Results
| Model | mAP50 | mAP50-95 |
|-------|-------|----------|
| YOLOv8n (ours) | 0.887 | 0.641 |

## 🛠️ Tech Stack
- YOLOv8 — Object Detection
- KITTI Dataset — 7481 real driving images
- PyTorch

## 📁 Dataset
KITTI Object Detection Dataset
- 7481 training images
- 5 classes: Car, Pedestrian, Cyclist, Van, Truck

## 🚀 How to Run

### 1. Install dependencies
pip install ultralytics

### 2. Train
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(data='kitti.yaml', epochs=50)

### 3. Inference
model = YOLO('best.pt')
results = model('image.png')
results[0].show()

## 📈 Coming Soon
- Semantic Segmentation (SegFormer)
- Monocular Depth Estimation
- Multi-Object Tracking
