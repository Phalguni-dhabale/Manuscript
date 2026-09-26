"""
Feature extraction module for Manuscript Layout Region Detection.
Implements Stage 3: Geometric and visual feature vector computation per candidate region box.
"""

import numpy as np
from typing import List, Tuple, Dict, Any

FEATURE_NAMES = [
    'rel_y_min',
    'rel_y_max',
    'rel_y_center',
    'rel_x_min',
    'rel_x_max',
    'rel_x_center',
    'rel_width',
    'rel_height',
    'rel_area',
    'aspect_ratio',
    'dist_top',
    'dist_bottom',
    'dist_left',
    'dist_right',
    'min_edge_dist',
    'text_density'
]

def extract_box_features(
    box: Tuple[int, int, int, int],
    page_shape: Tuple[int, int],
    binary_mask: np.ndarray = None
) -> np.ndarray:
    """
    Computes a 16-dimensional geometric and visual feature vector for a single bounding box.
    
    Parameters:
    - box: (x1, y1, x2, y2)
    - page_shape: (height, width)
    - binary_mask: optional 2D binary image mask for foreground text density
    
    Returns:
    - 1D numpy array of float32 features
    """
    H, W = page_shape[:2]
    H = max(1, H)
    W = max(1, W)
    
    x1, y1, x2, y2 = box
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    
    rel_y_min = y1 / float(H)
    rel_y_max = y2 / float(H)
    rel_y_center = (y1 + y2) / (2.0 * H)
    
    rel_x_min = x1 / float(W)
    rel_x_max = x2 / float(W)
    rel_x_center = (x1 + x2) / (2.0 * W)
    
    rel_width = bw / float(W)
    rel_height = bh / float(H)
    rel_area = (bw * bh) / float(W * H)
    
    aspect_ratio = bw / float(bh)
    
    dist_top = rel_y_min
    dist_bottom = (H - y2) / float(H)
    dist_left = rel_x_min
    dist_right = (W - x2) / float(W)
    min_edge_dist = min(dist_top, dist_bottom, dist_left, dist_right)
    
    # Text stroke density inside box crop
    text_density = 0.5  # default
    if binary_mask is not None and binary_mask.shape[0] == H and binary_mask.shape[1] == W:
        crop = binary_mask[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
        if crop.size > 0:
            text_density = float(np.count_nonzero(crop > 0)) / float(crop.size)
            
    features = np.array([
        rel_y_min,
        rel_y_max,
        rel_y_center,
        rel_x_min,
        rel_x_max,
        rel_x_center,
        rel_width,
        rel_height,
        rel_area,
        aspect_ratio,
        dist_top,
        dist_bottom,
        dist_left,
        dist_right,
        min_edge_dist,
        text_density
    ], dtype=np.float32)
    
    return features

def extract_batch_features(
    boxes: List[Tuple[int, int, int, int]],
    page_shape: Tuple[int, int],
    binary_mask: np.ndarray = None
) -> np.ndarray:
    """
    Extracts feature matrix (N, 16) for a list of bounding boxes.
    """
    if not boxes:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
        
    feature_list = [extract_box_features(box, page_shape, binary_mask) for box in boxes]
    return np.vstack(feature_list)
