from ultralytics import YOLO
from pathlib import Path

model_path = r"C:\Users\Mobina\PyCharmMiscProject\Perception-Autonomous-Driving-Improved\runs\detect\kitti_yolov8\weights\best.pt"

print(f"Model path: {model_path}")
print(f"Exists: {Path(model_path).exists()}")

model = YOLO(model_path)

print("Exporting to ONNX...")
model.export(format="onnx", imgsz=640, dynamic=True, simplify=True, opset=12)
print("✅ Done!")