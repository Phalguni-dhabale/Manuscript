"""
Visualization module for Manuscript Layout Region Detection.
Draws color-coded bounding boxes, class labels, and confidence tags on manuscript copy.
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Any
from src.utils import logger

# Class color map (BGR format for OpenCV)
CLASS_COLOR_MAP_BGR = {
    'header': (71, 99, 255),      # Coral Red
    'footer': (220, 180, 0),      # Cyan / Teal
    'main_text': (46, 204, 113),   # Emerald Green
    'side_text': (182, 89, 155),   # Amethyst Purple
    'filler': (18, 156, 243)       # Gold / Amber
}

DEFAULT_COLOR = (128, 128, 128)

def render_annotated_image(
    image: np.ndarray,
    predictions: List[Dict[str, Any]],
    thickness: int = 2
) -> np.ndarray:
    """
    Renders annotated boxes, labels, and confidence scores onto an image copy.
    """
    canvas = image.copy()
    h, w = canvas.shape[:2]
    
    # Scale thickness based on image resolution
    scale_factor = max(1.0, min(w, h) / 800.0)
    line_thickness = int(max(2, round(3 * scale_factor)))
    font_scale = max(0.55, 0.6 * scale_factor)
    
    for pred in predictions:
        box = pred['box']
        label = pred['label']
        score = pred['score']
        
        x1, y1, x2, y2 = box
        color = CLASS_COLOR_MAP_BGR.get(label, DEFAULT_COLOR)
        
        # Draw bounding box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, line_thickness)
        
        # Prepare label tag
        text = f"{label} {score:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        
        # Position label banner above or inside box
        banner_y1 = max(0, y1 - text_h - 6)
        banner_y2 = y1 if y1 - text_h - 6 >= 0 else y1 + text_h + 6
        
        # Draw solid background badge for tag text readability
        cv2.rectangle(canvas, (x1, banner_y1), (x1 + text_w + 8, banner_y2), color, -1)
        
        # Text color: white
        text_y = banner_y2 - 3 if banner_y2 == y1 else banner_y2 - 3
        cv2.putText(
            canvas,
            text,
            (x1 + 4, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
        
    return canvas

def save_annotated_image(output_path: str, canvas: np.ndarray) -> None:
    """Saves rendered canvas image to output path."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, canvas)
    logger.info(f"Saved annotated image to {output_path}")
