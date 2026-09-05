import cv2
import numpy as np
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ColorDetector:
    def __init__(self):
        pass
    
    def detect_color(self, vehicle_roi: np.ndarray) -> Tuple[str, float]:
        if vehicle_roi is None or vehicle_roi.size == 0:
            return 'Unknown', 0.0
        
        try:
            hsv = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2HSV)
        except:
            return 'Unknown', 0.0
        
        mean_hsv = np.mean(hsv, axis=(0, 1))
        h, s, v = mean_hsv
        
        if v < 40:
            return 'Black', 0.9
        if v > 200 and s < 50:
            return 'White', 0.85
        if s < 30:
            return 'Gray', 0.8
        
        if 0 <= h < 10 or 170 <= h <= 180:
            return 'Red', 0.9
        elif 10 <= h < 25:
            return 'Orange', 0.85
        elif 25 <= h < 40:
            return 'Yellow', 0.85
        elif 40 <= h < 80:
            return 'Green', 0.9
        elif 100 <= h < 130:
            return 'Blue', 0.9
        
        return 'Unknown', 0.5