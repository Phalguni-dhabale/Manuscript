"""
Utility functions for Manuscript Layout Region Detection pipeline.
Handles file IO, directory scanning, JSON result serialization, and logging.
"""

import os
import json
import logging
from typing import List, Dict, Any, Tuple
import cv2
import numpy as np

def setup_logger(name: str = "manuscript_layout", level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a standard logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger

logger = setup_logger()

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

def collect_image_paths(input_path: str) -> List[str]:
    """
    Collects image file paths from a single file path or a directory path.
    Supports relative and absolute paths.
    """
    abs_path = os.path.abspath(input_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    
    if os.path.isfile(abs_path):
        ext = os.path.splitext(abs_path)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            return [abs_path]
        else:
            raise ValueError(f"Unsupported file format: {ext}. Supported: {SUPPORTED_EXTENSIONS}")
            
    image_paths = []
    for root, _, files in os.walk(abs_path):
        for file in sorted(files):
            ext = os.path.splitext(file)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                image_paths.append(os.path.join(root, file))
                
    if not image_paths:
        logger.warning(f"No valid images found under: {input_path}")
        
    return sorted(image_paths)

def load_image(image_path: str) -> Tuple[np.ndarray, int, int]:
    """
    Loads an image from path using OpenCV.
    Returns (bgr_image, height, width).
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
        
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from: {image_path}")
        
    h, w = image.shape[:2]
    return image, h, w

def save_json_results(output_path: str, data: Dict[str, Any]) -> None:
    """
    Saves metadata prediction dictionary as formatted JSON.
    Ensures parent directories exist.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved predictions JSON to {output_path}")
