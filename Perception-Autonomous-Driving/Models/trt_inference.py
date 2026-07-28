# models/trt_inference.py
import os
import sys

# ۱. اضافه کردن مسیر DLLهای CUDAی موجود در PyTorch به PATH سیستم
try:
    import torch

    torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")
    if os.path.exists(torch_lib_path):
        os.add_dll_directory(torch_lib_path)
        os.environ["PATH"] = torch_lib_path + os.path.pathsep + os.environ["PATH"]
except Exception as e:
    print(f"Warning loading torch DLLs: {e}")

# ۲. حالا وارد کردن onnxruntime
import onnxruntime as ort
import numpy as np


def get_available_providers():
    """تشخیص خودکار بهترین Provider موجود روی سیستم"""
    available = ort.get_available_providers()
    providers = []

    if 'TensorRTExecutionProvider' in available:
        providers.append(('TensorRTExecutionProvider', {
            'device_id': 0,
            'trt_fp16_enable': True,
            'trt_max_workspace_size': 2147483648,
        }))
    if 'CUDAExecutionProvider' in available:
        providers.append('CUDAExecutionProvider')

    providers.append('CPUExecutionProvider')
    return providers


class FastYOLO:
    """کلاس اجرای YOLOv8 بهینه‌شده"""

    def __init__(self, onnx_path="yolov8n.onnx"):
        self.session = ort.InferenceSession(onnx_path, providers=get_available_providers())
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, input_tensor_np):
        return self.session.run(None, {self.input_name: input_tensor_np})


class FastMiDaS:
    """کلاس اجرای MiDaS بهینه‌شده"""

    def __init__(self, onnx_path="weights/midas_small.onnx"):
        self.session = ort.InferenceSession(onnx_path, providers=get_available_providers())
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, input_tensor_np):
        return self.session.run(None, {self.input_name: input_tensor_np})