import cv2
import numpy as np
import threading
import time
from typing import Optional, List, Tuple
from queue import Queue
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logger.warning("EasyOCR не встановлено")


class PlateReader:
    def __init__(self, plate_model_path=None, confidence_threshold=0.5, use_gpu=False):
        self.confidence_threshold = confidence_threshold
        
        self.ocr = None
        self.ocr_queue = Queue()
        self.ocr_results = {}
        self.ocr_thread = None
        self.ocr_running = False
        
        if EASYOCR_AVAILABLE:
            self._init_ocr()
    
    def _init_ocr(self):
        try:
            self.ocr = easyocr.Reader(['en'], gpu=False)
            logger.info("EasyOCR ініціалізовано")
            
            self.ocr_running = True
            self.ocr_thread = threading.Thread(target=self._ocr_loop, daemon=True)
            self.ocr_thread.start()
        except Exception as e:
            logger.error(f"Помилка OCR: {e}")
            self.ocr = None
    
    def _ocr_loop(self):
        while self.ocr_running:
            try:
                if not self.ocr_queue.empty():
                    plate_id, image = self.ocr_queue.get(timeout=0.1)
                    if plate_id is None:
                        break
                    
                    if self.ocr and image is not None and image.size > 0:
                        try:
                            processed = self._preprocess(image)
                            result = self.ocr.readtext(processed, detail=0, paragraph=False)
                            if result:
                                text = ''.join(result).upper()
                                text = ''.join(c for c in text if c.isalnum())
                                if text and len(text) >= 3:
                                    self.ocr_results[plate_id] = {
                                        'text': text,
                                        'timestamp': time.time()
                                    }
                                    logger.info(f"Номер: {text}")
                        except Exception as e:
                            logger.error(f"Помилка OCR: {e}")
                    
                    self.ocr_queue.task_done()
                else:
                    current_time = time.time()
                    expired = [k for k, v in self.ocr_results.items() 
                              if current_time - v['timestamp'] > 5.0]
                    for k in expired:
                        del self.ocr_results[k]
                    time.sleep(0.01)
            except:
                time.sleep(0.1)
    
    def _preprocess(self, image):
        if image is None or image.size == 0:
            return np.zeros((96, 96), dtype=np.uint8)
        
        h, w = image.shape[:2]
        if h > 96:
            scale = 96 / h
            new_w = int(w * scale)
            image = cv2.resize(image, (new_w, 96))
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    
    def detect_plates(self, vehicle_roi: np.ndarray) -> List[Tuple[int, int, int, int]]:
        plates = []
        
        if vehicle_roi is None or vehicle_roi.size == 0:
            return plates
        
        try:
            h, w = vehicle_roi.shape[:2]
            gray = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2GRAY)
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            
            edges = cv2.Canny(gray, 50, 150)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            for cnt in contours[:15]:
                x, y, cw, ch = cv2.boundingRect(cnt)
                
                if cw < 30 or ch < 10 or cw > w * 0.8 or ch > h * 0.5:
                    continue
                
                aspect_ratio = cw / ch if ch > 0 else 0
                area_ratio = (cw * ch) / (w * h)
                
                if 2.0 < aspect_ratio < 5.0 and area_ratio > 0.01:
                    padding = 5
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(w, x + cw + padding)
                    y2 = min(h, y + ch + padding)
                    plates.append((x1, y1, x2, y2))
                    
                    if len(plates) >= 2:
                        break
                        
        except Exception as e:
            logger.error(f"Помилка детекції: {e}")
        
        return plates
    
    def read_plate(self, plate_id: int, image: np.ndarray) -> Optional[str]:
        if self.ocr is None or image is None or image.size == 0:
            return None
        
        self.ocr_queue.put((plate_id, image))
        
        timeout = 2.0
        start = time.time()
        while time.time() - start < timeout:
            if plate_id in self.ocr_results:
                result = self.ocr_results[plate_id]
                del self.ocr_results[plate_id]
                return result['text']
            time.sleep(0.05)
        return None
    
    def shutdown(self):
        self.ocr_running = False
        if self.ocr_queue:
            self.ocr_queue.put((None, None))
        if self.ocr_thread:
            self.ocr_thread.join(timeout=2.0)