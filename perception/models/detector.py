import torch
from ultralytics import YOLO
from pathlib import Path
import yaml

class Detector:
    def __init__(self, config_path="perception/config/config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.device = self.config['inference']['device']
        self.model_path = self.config['model']['detector']['path']
        
        print(f"Loading YOLOv8 detector from {self.model_path}...")
        self.model = YOLO(self.model_path)
        self.model.to(self.device)
        print("Detector loaded successfully!")
    
    def detect(self, image):
        """Run detection on image"""
        results = self.model(image, conf=self.config['model']['detector']['conf'], 
                           iou=self.config['model']['detector']['iou'])
        return results[0]
    
    def get_boxes(self, result):
        """Extract boxes info"""
        boxes = []
        for box in result.boxes:
            boxes.append({
                "class": result.names[int(box.cls)],
                "confidence": round(float(box.conf), 3),
                "bbox": [round(x) for x in box.xyxy[0].tolist()]
            })
        return boxes