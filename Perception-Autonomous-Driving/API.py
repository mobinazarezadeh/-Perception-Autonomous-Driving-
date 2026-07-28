# API.py
import os
import sys
import importlib.util
from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np

# ۱. بارگذاری ایمن کلاس‌های FastYOLO و FastMiDaS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
trt_path = os.path.join(BASE_DIR, "models", "trt_inference.py")

spec = importlib.util.spec_from_file_location("trt_inference", trt_path)
trt_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trt_module)

FastYOLO = trt_module.FastYOLO
FastMiDaS = trt_module.FastMiDaS

# ۲. راه‌اندازی FastAPI
app = FastAPI(title="Fast Autonomous Driving Perception API")

# ۳. بارگذاری مدل‌های بهینه‌شده
print("Loading ONNX / GPU Models...")
yolo_model = FastYOLO("yolov8n.onnx")
midas_model = FastMiDaS("weights/midas_small.onnx")
print("✓ Models Loaded Successfully on GPU!")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # خواندن تصویر ورودی
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # پیش‌پردازش برای YOLO (تغییر سایز به 640x640)
    img_yolo = cv2.resize(img, (640, 640))
    img_yolo = img_yolo.transpose((2, 0, 1)) / 255.0
    img_yolo = np.expand_dims(img_yolo, axis=0).astype(np.float32)

    # پیش‌پردازش برای MiDaS (تغییر سایز به 256x256)
    img_midas = cv2.resize(img, (256, 256))
    img_midas = img_midas.transpose((2, 0, 1)) / 255.0
    img_midas = np.expand_dims(img_midas, axis=0).astype(np.float32)

    # استنتاج سریع روی GPU
    detections = yolo_model.predict(img_yolo)
    depth_map = midas_model.predict(img_midas)

    return {
        "status": "success",
        "detections_shape": list(detections[0].shape),
        "depth_map_shape": list(depth_map[0].shape),
        "message": "Inference executed on RTX GPU with ONNX Runtime",
    }


