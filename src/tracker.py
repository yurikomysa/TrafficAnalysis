from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VehicleTracker:
    def __init__(self, max_age=30, n_init=3, max_iou_distance=0.7, **kwargs):
        self.max_age = max_age
        self.min_hits = n_init
        self.iou_threshold = max_iou_distance
        self.tracks = {}
        self.next_id = 0
        self.frame_count = 0
        logger.info("Трекер ініціалізовано")
    
    def update(self, detections: List[Dict], frame=None) -> List[Dict]:
        self.frame_count += 1
        
        for track_id in list(self.tracks.keys()):
            self.tracks[track_id]['age'] += 1
            if self.tracks[track_id]['age'] > self.max_age:
                del self.tracks[track_id]
        
        if not detections:
            return self._get_active()
        
        matched, unmatched_dets, unmatched_tracks = self._match(detections)
        results = []
        
        for track_id, det_idx in matched:
            det = detections[det_idx]
            self.tracks[track_id]['bbox'] = det['bbox']
            self.tracks[track_id]['age'] = 0
            self.tracks[track_id]['hits'] += 1
            self.tracks[track_id]['confidence'] = det['confidence']
            self.tracks[track_id]['class_id'] = det['class_id']
            self.tracks[track_id]['class_name'] = det['class_name']
            
            if self.tracks[track_id]['hits'] >= self.min_hits:
                results.append(self._to_dict(track_id))
        
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            track_id = self.next_id
            self.next_id += 1
            
            self.tracks[track_id] = {
                'bbox': det['bbox'],
                'age': 0,
                'hits': 1,
                'confidence': det['confidence'],
                'class_id': det['class_id'],
                'class_name': det['class_name']
            }
            
            if self.tracks[track_id]['hits'] >= self.min_hits:
                results.append(self._to_dict(track_id))
        
        return results
    
    def _match(self, detections):
        track_ids = list(self.tracks.keys())
        if not track_ids:
            return [], list(range(len(detections))), []
        
        iou_matrix = self._compute_iou(detections, track_ids)
        matched = []
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = track_ids.copy()
        
        while unmatched_dets and unmatched_tracks:
            best_iou = -1
            best_det = -1
            best_track = -1
            
            for i, det_idx in enumerate(unmatched_dets):
                for j, track_id in enumerate(unmatched_tracks):
                    iou = iou_matrix[i][j]
                    if iou > best_iou and iou > self.iou_threshold:
                        best_iou = iou
                        best_det = i
                        best_track = j
            
            if best_det >= 0 and best_track >= 0:
                matched.append((unmatched_tracks.pop(best_track), unmatched_dets.pop(best_det)))
            else:
                break
        
        return matched, unmatched_dets, unmatched_tracks
    
    def _compute_iou(self, detections, track_ids):
        matrix = []
        for det in detections:
            row = []
            for track_id in track_ids:
                row.append(self._iou(det['bbox'], self.tracks[track_id]['bbox']))
            matrix.append(row)
        return matrix
    
    def _iou(self, b1, b2):
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2-x1) * (y2-y1)
        area1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
        area2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0
    
    def _get_active(self):
        results = []
        for track_id, track in self.tracks.items():
            if track['hits'] >= self.min_hits:
                results.append(self._to_dict(track_id))
        return results
    
    def _to_dict(self, track_id):
        track = self.tracks[track_id]
        return {
            'track_id': track_id,
            'bbox': track['bbox'],
            'confidence': track.get('confidence', 0.5),
            'class_id': track.get('class_id', 2),
            'class_name': track.get('class_name', 'car'),
            'age': track['age']
        }
    
    def get_active_tracks(self):
        return self._get_active()