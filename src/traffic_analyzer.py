import cv2
import numpy as np
import threading
import time
import logging
from typing import Optional, Dict, List, Any
from queue import Queue

from .stream_reader import StreamReader
from .vehicle_detector import VehicleDetector
from .tracker import VehicleTracker
from .plate_reader import PlateReader
from .color_detector import ColorDetector
from .utils import draw_vehicle_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrafficAnalyzer:
    def __init__(self, rtsp_url: str = None, vehicle_model: str = 'yolov8n.pt',
                 confidence: float = 0.5, device: str = 'cpu', skip_frames: int = 1):
        self.rtsp_url = rtsp_url
        self.confidence = confidence
        self.device = device
        self.skip_frames = skip_frames
        
        self.frame_count = 0
        self.detected_vehicles = 0
        self.plates_detected = 0
        self.vehicle_info = {}
        self.prev_tracks = {}
        
        self.fps = 0
        self.last_fps_update = time.time()
        self.fps_counter = 0
        
        self.running = False
        self.thread = None
        self.result_queue = Queue(maxsize=10)
        
        logger.info("Initializing Traffic Analyzer...")
        
        self.stream_reader = None
        if rtsp_url:
            self.stream_reader = StreamReader(rtsp_url)
        
        self.vehicle_detector = VehicleDetector(vehicle_model, confidence, device)
        self.tracker = VehicleTracker()
        self.color_detector = ColorDetector()
        
        try:
            self.plate_reader = PlateReader(
                confidence_threshold=confidence,
                use_gpu=(device == 'cuda')
            )
            logger.info("Plate reader initialized")
        except Exception as e:
            logger.error(f"Failed to initialize plate reader: {e}")
            self.plate_reader = None
        
        logger.info("Traffic Analyzer ready")
    
    def start_processing(self, rtsp_url: str = None):
        if rtsp_url:
            self.rtsp_url = rtsp_url
            self.stream_reader = StreamReader(rtsp_url)
        
        if not self.stream_reader:
            logger.error("No stream reader available")
            return False
        
        if self.running:
            return True
        
        self.running = True
        self.thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.thread.start()
        logger.info("Processing started")
        return True
    
    def stop_processing(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=3.0)
        if self.stream_reader:
            self.stream_reader.stop()
        if self.plate_reader:
            self.plate_reader.shutdown()
        logger.info("Processing stopped")
    
    def _processing_loop(self):
        if not self.stream_reader.start():
            logger.error("Failed to start stream reader")
            self.running = False
            return
        
        logger.info("Processing loop started")
        
        while self.running:
            try:
                frame = self.stream_reader.get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                self.frame_count += 1
                
                result = self._process_frame(frame)
                
                self.fps_counter += 1
                current_time = time.time()
                if current_time - self.last_fps_update >= 1.0:
                    self.fps = self.fps_counter
                    self.fps_counter = 0
                    self.last_fps_update = current_time
                
                try:
                    if self.result_queue.full():
                        self.result_queue.get_nowait()
                    self.result_queue.put_nowait(result)
                except:
                    pass
                
            except Exception as e:
                logger.error(f"Processing error: {e}")
                time.sleep(0.1)
        
        logger.info("Processing loop stopped")
    
    def _process_frame(self, frame):
        processed = frame.copy()
        
        do_detection = (self.frame_count % (self.skip_frames + 1) == 0)
        
        if do_detection:
            detections = self.vehicle_detector.detect_vehicles(frame)
            if detections:
                tracks = self.tracker.update(detections, frame)
                self.prev_tracks = {t['track_id']: t for t in tracks}
            else:
                tracks = self.tracker.update([], frame)
                self.prev_tracks = {t['track_id']: t for t in tracks}
        else:
            tracks = list(self.prev_tracks.values())
        
        for track in tracks:
            track_id = track['track_id']
            bbox = track['bbox']
            
            roi = self.vehicle_detector.get_vehicle_roi(frame, bbox)
            color_name, _ = self.color_detector.detect_color(roi)
            
            plate_text = None
            if self.plate_reader and roi is not None and roi.size > 0:
                plate_boxes = self.plate_reader.detect_plates(roi)
                if plate_boxes:
                    x1, y1, x2, y2 = plate_boxes[0]
                    plate_img = roi[y1:y2, x1:x2]
                    if plate_img.size > 0:
                        plate_text = self.plate_reader.read_plate(track_id, plate_img)
                        if plate_text:
                            self.plates_detected += 1
            
            self.vehicle_info[track_id] = {
                'track_id': track_id,
                'bbox': bbox,
                'color': color_name,
                'plate': plate_text or "",
                'confidence': track.get('confidence', 0.0)
            }
            
            draw_vehicle_info(processed, bbox, track_id, plate_text or "", color_name, track.get('confidence', 0.0))
        
        self.detected_vehicles = len(tracks)
        
        info_text = f"FPS: {self.fps:.1f} | Cars: {self.detected_vehicles} | Plates: {self.plates_detected}"
        cv2.putText(processed, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        _, jpeg = cv2.imencode('.jpg', processed)
        frame_bytes = jpeg.tobytes()
        
        return {
            'frame': frame_bytes,
            'vehicles': list(self.vehicle_info.values()),
            'count': len(tracks),
            'plates_detected': self.plates_detected,
            'fps': self.fps
        }
    
    def get_frame(self):
        try:
            if not self.result_queue.empty():
                return self.result_queue.get_nowait()
        except:
            pass
        return None
    
    def get_stats(self):
        """Отримання статистики"""
        # Конвертуємо всі значення в стандартні Python типи
        vehicles_list = []
        for v in self.vehicle_info.values():
            vehicles_list.append({
                'track_id': int(v.get('track_id', 0)),
                'color': str(v.get('color', 'Unknown')),
                'plate': str(v.get('plate', '')),
                'confidence': float(v.get('confidence', 0.0))
            })
        
        return {
            'total_frames': int(self.frame_count),
            'current_fps': float(self.fps),
            'vehicles_detected': int(self.detected_vehicles),
            'plates_detected': int(self.plates_detected),
            'active_vehicles': int(len(self.vehicle_info)),
            'vehicles': vehicles_list
        }