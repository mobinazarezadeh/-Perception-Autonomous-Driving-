# utils/exporter.py
import torch
from ultralytics import YOLO


def export_yolo_to_trt(yolo_weights_path: str, output_dir: str):
    """تبدیل YOLOv8 به فرمت ONNX بهینه‌شده برای TensorRT"""
    print("[1/3] Converting YOLOv8 to Optimized ONNX (TensorRT Ready)...")
    model = YOLO(yolo_weights_path)
    model.export(format='onnx', half=True, dynamic=True, simplify=True, device=0)
    print("✓ YOLOv8 ONNX export completed.")


def export_midas_to_onnx(onnx_output_path: str):
    """تبدیل مدل MiDaS به فرمت ONNX"""
    print("[2/3] Exporting MiDaS to ONNX...")

    # اصلاح آدرس ریپوزیتوری به isl-org/MiDaS و اضافه کردن trust_repo
    model = torch.hub.load("isl-org/MiDaS", "MiDaS_small", trust_repo=True)
    model.eval().cuda()

    dummy_input = torch.randn(1, 3, 256, 256, device='cuda')

    torch.onnx.export(
        model,
        dummy_input,
        onnx_output_path,
        opset_version=14,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    print("✓ MiDaS ONNX export completed.")