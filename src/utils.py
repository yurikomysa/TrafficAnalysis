import cv2
import numpy as np
from typing import Tuple


def draw_text_with_background(img, text, position, color=(0, 255, 0), bg_color=(0, 0, 0)):
    x, y = position
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    
    cv2.rectangle(img, (x - 3, y - text_h - 3), (x + text_w + 3, y + 3), bg_color, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img


def get_color_rgb(color_name):
    colors = {
        "Red": (0, 0, 255),
        "Blue": (255, 0, 0),
        "Green": (0, 255, 0),
        "Yellow": (0, 255, 255),
        "White": (255, 255, 255),
        "Black": (0, 0, 0),
        "Gray": (128, 128, 128),
        "Orange": (0, 165, 255),
        "Unknown": (128, 128, 128)
    }
    return colors.get(color_name, (128, 128, 128))


def draw_vehicle_info(frame, bbox, track_id, plate_text, color_name, confidence):
    x1, y1, x2, y2 = bbox
    color_rgb = get_color_rgb(color_name)
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), color_rgb, 2)
    
    text = f"ID:{track_id} | {color_name}"
    if plate_text:
        text += f" | {plate_text}"
    
    draw_text_with_background(frame, text, (x1, y1 - 10), color=color_rgb)
    return frame