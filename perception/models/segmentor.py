import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import yaml

class Segmentor:
    def __init__(self, config_path="perception/config/config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.device = self.config['inference']['device']
        self.model_name = self.config['model']['segmentor']['name']
        
        print(f"Loading SegFormer model: {self.model_name}...")
        self.processor = SegformerImageProcessor.from_pretrained(self.model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(self.model_name)
        self.model.to(self.device)
        print("Segmentor loaded successfully!")
    
    def segment(self, image):
        """Run semantic segmentation"""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Upsample to original size
        upsampled = torch.nn.functional.interpolate(
            outputs.logits, 
            size=image.size[::-1],
            mode='bilinear', 
            align_corners=False
        )
        seg_map = upsampled.argmax(dim=1)[0].cpu().numpy()
        
        classes_found = [int(x) for x in sorted(list(set(seg_map.flatten())))]
        
        return seg_map, classes_found