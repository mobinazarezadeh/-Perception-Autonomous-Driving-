# export_all.py
import sys
import os
import importlib.util

# پیدا کردن مسیر دقیق پوشه جاری فایل
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# افزودن صریح مسیر پروژه به sys.path
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# بارگذاری مستقیم فایل exporter.py بدون وابستگی به ساختار پکیج
exporter_path = os.path.join(BASE_DIR, "utils", "exporter.py")

if not os.path.exists(exporter_path):
    raise FileNotFoundError(f"فایل پیدا نشد! لطفا مطمئن شوید فایل در مسیر زیر وجود دارد:\n{exporter_path}")

spec = importlib.util.spec_from_file_location("exporter", exporter_path)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)

# فراخوانی توابع
export_yolo_to_trt = exporter.export_yolo_to_trt
export_midas_to_onnx = exporter.export_midas_to_onnx


def main():
    os.makedirs(os.path.join(BASE_DIR, "weights"), exist_ok=True)

    # ۱. تبدیل YOLO
    export_yolo_to_trt(yolo_weights_path="yolov8n.pt", output_dir="weights")

    # ۲. تبدیل MiDaS به ONNX
    export_midas_to_onnx(onnx_output_path=os.path.join(BASE_DIR, "weights", "midas_small.onnx"))

    print("\nAll exports finished! Check the 'weights' folder.")


if __name__ == "__main__":
    main()