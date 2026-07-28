from ultralytics import YOLO
import time
import numpy as np
from PIL import Image

model = YOLO("runs/detect/kitti_yolov8/weights/best.pt")

img = Image.open("pipeline_demo.png")

print("Warming up...")
for _ in range(20):
    _ = model(img, verbose=False)

print("Benchmarking...")
times = []
for i in range(50):
    start = time.time()
    _ = model(img, verbose=False)
    end = time.time()
    times.append(end - start)

print(f"Average time: {np.mean(times)*1000:.1f} ms")
print(f"FPS: {1/np.mean(times):.1f} FPS")

print("Benchmark completed!")