from perception.models.detector import Detector
from perception.models.segmentor import Segmentor
from PIL import Image
import time

class PerceptionPipeline:
    def __init__(self):
        self.detector = Detector()
        self.segmentor = Segmentor()
    
    def run(self, image: Image.Image):
        start = time.time()
        
        # Detection
        det_result = self.detector.detect(image)
        boxes = self.detector.get_boxes(det_result)
        
        # Segmentation
        seg_map, classes_found = self.segmentor.segment(image)
        
        elapsed = round((time.time() - start) * 1000, 1)
        
        return {
            "detection": {
                "objects": boxes,
                "total_objects": len(boxes)
            },
            "segmentation": {
                "classes_found": classes_found,
                "num_classes": len(classes_found)
            },
            "processing_time_ms": elapsed
        }