from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
import cv2
import numpy as np
from PIL import Image
import io
import time
from ultralytics import YOLO
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

app = FastAPI(title="Autonomous Driving Perception API")

#loading model while starting
print("loading model...")
detector = YOLO(r'C:\Users\Mobina\PyCharmMiscProject\runs\detect\kitti_yolov8\weights\best.pt')
processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-512-1024")
segmentor = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-512-1024")
print("Its ready")


@app.get("/")
def root():
    return {"message": "Autonomous Driving Perception API", "status": "running"}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    start = time.time()

    # loading image
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    img_np = np.array(img)

    # Detection
    results = detector(img_np, conf=0.5)
    boxes = results[0].boxes

    objects = []
    for box in boxes:
        objects.append({
            "class": detector.names[int(box.cls)],
            "confidence": round(float(box.conf), 3),
            "bbox": [round(x) for x in box.xyxy[0].tolist()]
        })

    elapsed = round((time.time() - start) * 1000, 1)

    return JSONResponse({
        "objects": objects,
        "total_objects": len(objects),
        "processing_time_ms": elapsed
    })


@app.post("/segment")
async def segment(file: UploadFile = File(...)):
    start = time.time()

    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')

    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = segmentor(**inputs)

    upsampled = torch.nn.functional.interpolate(
        outputs.logits, size=img.size[::-1],
        mode='bilinear', align_corners=False
    )
    seg_map = upsampled.argmax(dim=1)[0].numpy()

    classes_found = [int(x) for x in np.unique(seg_map).tolist()]
    elapsed = round((time.time() - start) * 1000, 1)

    return JSONResponse({
        "classes_found": classes_found,
        "num_classes": len(classes_found),
        "processing_time_ms": elapsed
    })


@app.post("/pipeline")
async def full_pipeline(file: UploadFile = File(...)):
    start = time.time()

    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    img_np = np.array(img)

    # Detection
    det_results = detector(img_np, conf=0.5)
    boxes = det_results[0].boxes
    objects = []
    for box in boxes:
        objects.append({
            "class": detector.names[int(box.cls)],
            "confidence": round(float(box.conf), 3),
            "bbox": [round(x) for x in box.xyxy[0].tolist()]
        })

    # Segmentation
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = segmentor(**inputs)
    upsampled = torch.nn.functional.interpolate(
        outputs.logits, size=img.size[::-1],
        mode='bilinear', align_corners=False
    )
    seg_map = upsampled.argmax(dim=1)[0].numpy()
    classes_found = [int(x) for x in np.unique(seg_map).tolist()]

    elapsed = round((time.time() - start) * 1000, 1)

    return JSONResponse({
        "detection": {
            "objects": objects,
            "total_objects": len(objects)
        },
        "segmentation": {
            "classes_found": classes_found,
            "num_classes": len(classes_found)
        },
        "processing_time_ms": elapsed
    })