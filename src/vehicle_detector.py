from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VehicleDetector:
    VEHICLE_CLASSES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
    
    def __init__(self, model_path='yolov8n.pt', confidence_threshold=0.5, device='cpu'):
        self.confidence_threshold = confidence_threshold
        self.model = YOLO(model_path)
        self.model.conf = confidence_threshold
        logger.info(f"Модель завантажено: {model_path}")
    
    def detect_vehicles(self, frame: np.ndarray) -> List[Dict]:
        results = self.model(frame, verbose=False)
        detections = []
        
        for r in results:
            if r.boxes:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    
                    if cls_id in self.VEHICLE_CLASSES and conf >= self.confidence_threshold:
                        detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'confidence': conf,
                            'class_id': cls_id,
                            'class_name': self.VEHICLE_CLASSES[cls_id]
                        })
        return detections
    
    def detect_vehicles_optimized(self, frame: np.ndarray, skip_frames=1) -> List[Dict]:
        if skip_frames > 1:
            if not hasattr(self, '_counter'):
                self._counter = 0
            self._counter += 1
            if self._counter % skip_frames != 0:
                return getattr(self, '_last', [])
        
        detections = self.detect_vehicles(frame)
        self._last = detections
        return detections
    
    def get_vehicle_roi(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        padding = 5
        return frame[max(0,y1-padding):y2+padding, max(0,x1-padding):x2+padding]