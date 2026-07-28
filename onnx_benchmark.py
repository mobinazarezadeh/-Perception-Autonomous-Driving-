from ultralytics import YOLO
import time
import numpy as np
from PIL import Image

# Export to ONNX
model = YOLO("runs/detect/kitti_yolov8/weights/best.pt")
print("Exporting to ONNX...")
model.export(format="onnx", imgsz=640, dynamic=True, simplify=True)

print("Benchmarking ONNX...")

# For now use PyTorch for benchmark, later ONNX Runtime
img = Image.open("pipeline_demo.png")
times = []
for i in range(100):
    start = time.time()
    _ = model(img, verbose=False)
    end = time.time()
    times.append(end - start)

print(f"Average time: {np.mean(times)*1000:.1f} ms")
print(f"FPS: {1/np.mean(times):.1f}")

print("Done!")