import cv2
import threading
import time
import logging
import os
from typing import Optional
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StreamReader:
    def __init__(self, rtsp_url: str, buffer_size: int = 2):
        self.url = rtsp_url
        self.buffer_size = buffer_size
        self.frame_buffer = []
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.cap = None
        self.fps = 0
        self.frame_count = 0
        self.last_time = time.time()
        
    def start(self) -> bool:
        if self.running:
            return True
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        return True
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
            
    def get_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            if not self.frame_buffer:
                return None
            return self.frame_buffer[-1].copy()
    
    def get_fps(self) -> float:
        return self.fps
    
    def _read_loop(self):
        while self.running:
            try:
                self._reconnect()
                
                while self.running and self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    
                    if not ret:
                        if self.url.startswith('file://') or os.path.exists(self.url.replace('file://', '')):
                            logger.info("Кінець відео, перемотуємо")
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                        else:
                            break
                    
                    with self.lock:
                        self.frame_buffer.append(frame)
                        if len(self.frame_buffer) > self.buffer_size:
                            self.frame_buffer.pop(0)
                    
                    self.frame_count += 1
                    current_time = time.time()
                    if current_time - self.last_time >= 1.0:
                        self.fps = self.frame_count
                        self.frame_count = 0
                        self.last_time = current_time
                    
                    time.sleep(0.001)
                    
            except Exception as e:
                logger.error(f"Помилка: {e}")
                time.sleep(2.0)
    
    def _reconnect(self):
        if self.cap:
            self.cap.release()
        
        if self.url.startswith('file://'):
            file_path = self.url[7:]
            if os.path.exists(file_path):
                self.cap = cv2.VideoCapture(file_path)
                logger.info(f"Відкрито файл: {file_path}")
            else:
                logger.error(f"Файл не знайдено: {file_path}")
                time.sleep(1.0)
                return
        else:
            self.cap = cv2.VideoCapture(self.url)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not self.cap.isOpened():
            logger.error(f"Не вдалося відкрити: {self.url}")
            time.sleep(1.0)
        else:
            logger.info(f"Підключено: {self.url}")