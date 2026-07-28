# test_speed.py
import time
import numpy as np
import os
import sys
import importlib.util

# پیدا کردن مسیر دقیق پوشه پروژه
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# بارگذاری مستقیم فایل trt_inference.py جهت جلوگیری از تداخل نام پوشه models
trt_path = os.path.join(BASE_DIR, "models", "trt_inference.py")

if not os.path.exists(trt_path):
    raise FileNotFoundError(f"فایل پیدا نشد! لطفا بررسی کنید که فایل در این مسیر وجود دارد:\n{trt_path}")

spec = importlib.util.spec_from_file_location("trt_inference", trt_path)
trt_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trt_module)

FastYOLO = trt_module.FastYOLO
FastMiDaS = trt_module.FastMiDaS


def benchmark():
    print("--- Starting Speed Test ---")

    # ۱. بارگذاری مدل‌ها
    yolo_model = FastYOLO("yolov8n.onnx")
    midas_model = FastMiDaS("weights/midas_small.onnx")

    # ۲. ورودی‌های نمونه (Dummy Inputs) با نوع float16 برای مطابقت با FP16
    dummy_img_yolo = np.random.randn(1, 3, 640, 640).astype(np.float16)
    dummy_img_midas = np.random.randn(1, 3, 256, 256).astype(np.float32)

    # ۳. گرم کردن مدل (Warm-up)
    print("Warming up models...")
    _ = yolo_model.predict(dummy_img_yolo)
    _ = midas_model.predict(dummy_img_midas)

    # ۴. تست سرعت YOLO
    start = time.time()
    for _ in range(50):
        _ = yolo_model.predict(dummy_img_yolo)
    yolo_time = (time.time() - start) / 50 * 1000

    # ۵. تست سرعت MiDaS
    start = time.time()
    for _ in range(50):
        _ = midas_model.predict(dummy_img_midas)
    midas_time = (time.time() - start) / 50 * 1000

    print("\n--- Results ---")
    print(f"✓ YOLOv8 Average Latency: {yolo_time:.2f} ms ({1000/yolo_time:.1f} FPS)")
    print(f"✓ MiDaS Average Latency:  {midas_time:.2f} ms ({1000/midas_time:.1f} FPS)")


if __name__ == "__main__":
    benchmark()
