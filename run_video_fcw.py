# run_video_fcw.py
import os
import cv2
import numpy as np
import importlib.util

# ۱. پیدا کردن هوشمند مسیر trt_inference.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
trt_path = None
for root, dirs, files in os.walk(BASE_DIR):
    if "trt_inference.py" in files:
        trt_path = os.path.join(root, "trt_inference.py")
        break

spec = importlib.util.spec_from_file_location("trt_inference", trt_path)
trt_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trt_module)

FastYOLO = trt_module.FastYOLO
FastMiDaS = trt_module.FastMiDaS


# ۲. سیستم هشدار برخورد (Forward Collision Warning)
class ForwardCollisionWarning:
    def __init__(self, warning_thresh=15.0, critical_thresh=0.2):
        self.warning_thresh = warning_thresh  # آستانه هشدار زرد (متر)
        self.critical_thresh = critical_thresh  # آستانه خطر قرمز و ترمز (متر)
        self.target_classes = [0, 2, 3, 5, 7]  # کلاس‌ها: عابر، ماشین، موتور، اتوبوس، کامیون

    def estimate_distance(self, depth_map_crop):
        if depth_map_crop.size == 0:
            return float('inf')

        # میانه مقادیر عمق داخل باکس
        median_depth = np.median(depth_map_crop)
        # فرمول نگاشت عمق MiDaS به فاصله واقعی (متر)
        distance_meters = 100.0 / (median_depth + 1e-5)
        return distance_meters

    def process_frame(self, frame, detections, depth_map, orig_w, orig_h):
        depth_h, depth_w = depth_map.shape
        has_critical_alert = False

        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det
            cls_id = int(cls_id)

            if cls_id not in self.target_classes:
                continue

            # تبدیل مختصات 640x640 به ابعاد ویدیو و ابعاد Depth
            ix1 = int(x1 * orig_w / 640)
            iy1 = int(y1 * orig_h / 640)
            ix2 = int(x2 * orig_w / 640)
            iy2 = int(y2 * orig_h / 640)

            dx1 = max(0, int(x1 * depth_w / 640))
            dy1 = max(0, int(y1 * depth_h / 640))
            dx2 = min(depth_w, int(x2 * depth_w / 640))
            dy2 = min(depth_h, int(y2 * depth_h / 640))

            depth_crop = depth_map[dy1:dy2, dx1:dx2]
            distance = self.estimate_distance(depth_crop)

            # تعیین وضعیت هشدار بر اساس فاصله
            if distance < self.critical_thresh:
                color = (0, 0, 255)  # قرمز: خطر برخورد شدید
                status_text = f"CRITICAL! {distance:.1f}m"
                has_critical_alert = True
            elif distance < self.warning_thresh:
                color = (0, 255, 255)  # زرد: هشدار نزدیک شدن
                status_text = f"Warning: {distance:.1f}m"
            else:
                color = (0, 255, 0)  # سبز: فاصله ایمن
                status_text = f"Safe: {distance:.1f}m"

            # رسم باکس و فاصله رو فریم
            cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), color, 2)
            cv2.putText(frame, status_text, (ix1, max(iy1 - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # نمایش پیام ترمز اضطراری در بالای تصویر
        if has_critical_alert:
            # کادر قرمز هشدار بالای ویدیو
            cv2.rectangle(frame, (0, 0), (orig_w, 60), (0, 0, 255), -1)
            cv2.putText(frame, "!!! CRITICAL WARNING: BRAKE NOW !!!", (orig_w // 8, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)

        return frame


# ۳. پست‌پراسس خروجی YOLO
def postprocess_yolo(output, conf_threshold=0.3):
    predictions = np.squeeze(output[0]).T
    boxes, confs, class_ids = [], [], []

    for pred in predictions:
        scores = pred[4:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > conf_threshold:
            cx, cy, w, h = pred[0:4]
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)

            boxes.append([x1, y1, x2, y2])
            confs.append(float(confidence))
            class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confs, conf_threshold, 0.4)
    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            results.append([*boxes[i], confs[i], class_ids[i]])
    return results


# ۴. پردازش اصلی ویدیو
def process_video(video_path, output_path="output_fcw.mp4"):
    yolo_path = "yolov8n.onnx"
    midas_path = "weights/midas_small.onnx"

    if not os.path.exists(yolo_path):
        for root, dirs, files in os.walk(BASE_DIR):
            if "yolov8n.onnx" in files: yolo_path = os.path.join(root, "yolov8n.onnx")
            if "midas_small.onnx" in files: midas_path = os.path.join(root, "midas_small.onnx")

    yolo = FastYOLO(yolo_path)
    midas = FastMiDaS(midas_path)
    fcw = ForwardCollisionWarning(warning_thresh=15.0, critical_thresh=8.0)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"خطا: فایل ویدیویی یافت نشد: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))

    print("--- پردازش ویدیو همراه با سیستم هشدار برخورد (FCW) آغاز شد... ---")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # پیش‌پردازش
        img_yolo = cv2.resize(frame, (640, 640)).transpose((2, 0, 1)) / 255.0
        img_yolo = np.expand_dims(img_yolo, axis=0).astype(np.float16)

        img_midas = cv2.resize(frame, (256, 256)).transpose((2, 0, 1)) / 255.0
        img_midas = np.expand_dims(img_midas, axis=0).astype(np.float32)

        # استنتاج مدل‌ها
        yolo_raw = yolo.predict(img_yolo)
        depth_raw = midas.predict(img_midas)[0].squeeze()

        # استخراج باکس‌ها
        detections = postprocess_yolo(yolo_raw)

        # پردازش هشدار برخورد روی فریم اصلی
        frame_processed = fcw.process_frame(frame, detections, depth_raw, width, height)

        # ساخت ویدیو Depth رنگی
        depth_normalized = cv2.normalize(depth_raw, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        depth_colored = cv2.resize(depth_colored, (width, height))

        # چسباندن تصویر پردازش‌شده و Depth کنار هم
        combined_frame = np.hstack((frame_processed, depth_colored))
        out.write(combined_frame)

    cap.release()
    out.release()
    print(f"✓ ویدیو خروجی همراه با پیام‌های هشدار در {output_path} ذخیره شد!")


if __name__ == "__main__":
    video_input = "sample_driving.mp4"

    if os.path.exists(video_input):
        process_video(video_input)
    else:
        print(f"لطفاً فایل ویدیو 'sample_driving.mp4' را در پوشه پروژه قرار دهید.")