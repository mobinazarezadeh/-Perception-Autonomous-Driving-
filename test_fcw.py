# test_fcw.py
import os
import sys
import cv2
import numpy as np
import importlib.util

# ۱. جستجوی هوشمند برای پیدا کردن مسیر دقیق trt_inference.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
trt_path = None

for root, dirs, files in os.walk(BASE_DIR):
    if "trt_inference.py" in files:
        trt_path = os.path.join(root, "trt_inference.py")
        break

if not trt_path or not os.path.exists(trt_path):
    raise FileNotFoundError("فایل trt_inference.py پیدا نشد!")

print(f"✓ Found inference script at: {trt_path}")

# بارگذاری دینامیک ماژول
spec = importlib.util.spec_from_file_location("trt_inference", trt_path)
trt_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trt_module)

FastYOLO = trt_module.FastYOLO
FastMiDaS = trt_module.FastMiDaS


# ۲. سیستم هشدار برخورد (Forward Collision Warning)
class ForwardCollisionWarning:
    def __init__(self, warning_thresh=15.0, critical_thresh=8.0):
        self.warning_thresh = warning_thresh  # آستانه هشدار (متر)
        self.critical_thresh = critical_thresh  # آستانه خطر و ترمز (متر)
        self.target_classes = [0, 2, 3, 5, 7]  # کلاس‌های هدف: انسان، ماشین، موتور، اتوبوس، کامیون

    def estimate_distance(self, depth_map_crop):
        """تخمین فاصله بر اساس میانه مقادیر عمق در ناحیه شیء"""
        if depth_map_crop.size == 0:
            return float('inf')

        median_depth = np.median(depth_map_crop)
        distance_meters = 100.0 / (median_depth + 1e-5)
        return distance_meters

    def process_frame(self, frame, detections, depth_map):
        h, w, _ = frame.shape
        depth_h, depth_w = depth_map.shape
        scale_x = depth_w / w
        scale_y = depth_h / h

        has_critical_alert = False

        for det in detections:
            if len(det) < 6:
                continue
            x1, y1, x2, y2, conf, cls_id = det[:6]
            cls_id = int(cls_id)

            if cls_id not in self.target_classes or conf < 0.25:
                continue

            ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
            dx1, dy1 = int(ix1 * scale_x), int(iy1 * scale_y)
            dx2, dy2 = int(ix2 * scale_x), int(iy2 * scale_y)

            depth_crop = depth_map[dy1:dy2, dx1:dx2]
            distance = self.estimate_distance(depth_crop)

            if distance < self.critical_thresh:
                color = (0, 0, 255)  # قرمز
                status_text = f"CRITICAL! {distance:.1f}m"
                has_critical_alert = True
            elif distance < self.warning_thresh:
                color = (0, 255, 255)  # زرد
                status_text = f"Warning: {distance:.1f}m"
            else:
                color = (0, 255, 0)  # سبز
                status_text = f"{distance:.1f}m"

            cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), color, 2)
            cv2.putText(frame, status_text, (ix1, max(iy1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if has_critical_alert:
            cv2.putText(frame, "WARNING: COLLISION RISK! BRAKE!", (w // 6, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

        return frame


# ۳. اجرای تست مدل
if __name__ == "__main__":
    print("--- Starting FCW Pipeline Test ---")

    yolo_path = "yolov8n.onnx"
    midas_path = "weights/midas_small.onnx"

    if not os.path.exists(yolo_path):
        for root, dirs, files in os.walk(BASE_DIR):
            if "yolov8n.onnx" in files:
                yolo_path = os.path.join(root, "yolov8n.onnx")
            if "midas_small.onnx" in files:
                midas_path = os.path.join(root, "midas_small.onnx")

    yolo = FastYOLO(yolo_path)
    midas = FastMiDaS(midas_path)
    fcw = ForwardCollisionWarning(warning_thresh=15.0, critical_thresh=8.0)

    # ساخت یک تصویر نمونه مشکی
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)

    # ۱. تنظیم ورودی YOLO به float16 (برای مطابقت با مدل FP16)
    img_yolo = cv2.resize(dummy_img, (640, 640)).transpose((2, 0, 1)) / 255.0
    img_yolo = np.expand_dims(img_yolo, axis=0).astype(np.float16)

    # ۲. تنظیم ورودی MiDaS به float32
    img_midas = cv2.resize(dummy_img, (256, 256)).transpose((2, 0, 1)) / 255.0
    img_midas = np.expand_dims(img_midas, axis=0).astype(np.float32)

    # استنتاج مدل‌ها
    dets_raw = yolo.predict(img_yolo)[0]
    depth_map = midas.predict(img_midas)[0].squeeze()

    print("✓ ONNX Models Inference Executed Successfully!")
    print(f"✓ YOLO Raw Output Shape: {dets_raw.shape}")
    print(f"✓ Depth Map Shape: {depth_map.shape}")
    print("✓ Forward Collision Warning (FCW) Module Ready!")