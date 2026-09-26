"""
Postprocessing module for Manuscript Layout Region Detection.
Implements Stage 4: Boundary clipping hard constraints, Non-Maximum Suppression (NMS),
confidence thresholding, containment suppression, and overlap resolution.

Key improvements over v1:
- Cross-class NMS (suppresses overlapping boxes of different classes)
- Containment suppression (removes small boxes fully inside larger boxes of same class)
- Higher minimum area threshold to suppress micro-fragments
- Deduplication pass for near-identical boxes
"""

from typing import List, Dict, Any, Tuple
import numpy as np

def clip_box_to_page_bounds(box: List[int], height: int, width: int) -> List[int]:
    """
    Hard constraint enforcer: Clips bounding box strictly within page boundaries.
    Ensures 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height.
    """
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, int(x1)))
    y1 = max(0, min(height - 1, int(y1)))
    x2 = max(x1 + 1, min(width, int(x2)))
    y2 = max(y1 + 1, min(height, int(y2)))
    return [x1, y1, x2, y2]

def compute_box_iou(b1: List[int], b2: List[int]) -> float:
    """Computes IoU between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h
    
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union_area = area1 + area2 - inter_area
    
    if union_area <= 0:
        return 0.0
    return inter_area / float(union_area)

def compute_containment(inner: List[int], outer: List[int]) -> float:
    """Returns the fraction of 'inner' box area that overlaps with 'outer' box."""
    x1 = max(inner[0], outer[0])
    y1 = max(inner[1], outer[1])
    x2 = min(inner[2], outer[2])
    y2 = min(inner[3], outer[3])
    
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h
    
    inner_area = max(1, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return inter_area / float(inner_area)

def apply_nms(
    predictions: List[Dict[str, Any]],
    iou_threshold: float = 0.45,
    class_agnostic: bool = False
) -> List[Dict[str, Any]]:
    """
    Applies Non-Maximum Suppression to predicted boxes.
    Sorts predictions by score descending and suppresses boxes with IoU > iou_threshold.
    """
    if not predictions:
        return []
        
    sorted_preds = sorted(predictions, key=lambda item: item['score'], reverse=True)
    keep = []
    
    for item in sorted_preds:
        box = item['box']
        label = item['label']
        suppress = False
        
        for k in keep:
            k_box = k['box']
            k_label = k['label']
            
            # If class agnostic or same class overlap
            if class_agnostic or (label == k_label):
                if compute_box_iou(box, k_box) > iou_threshold:
                    suppress = True
                    break
                    
        if not suppress:
            keep.append(item)
            
    return keep

def apply_containment_suppression(
    predictions: List[Dict[str, Any]],
    containment_threshold: float = 0.80
) -> List[Dict[str, Any]]:
    """
    Removes predictions where a smaller box is mostly contained inside a larger,
    higher-confidence box of the same class.
    """
    if len(predictions) <= 1:
        return predictions
    
    # Sort by area descending (largest first)
    sorted_preds = sorted(
        predictions, 
        key=lambda p: (p['box'][2] - p['box'][0]) * (p['box'][3] - p['box'][1]),
        reverse=True
    )
    
    keep = []
    for item in sorted_preds:
        is_contained = False
        for kept in keep:
            # Check if item is mostly inside kept
            if compute_containment(item['box'], kept['box']) > containment_threshold:
                # Only suppress if same class or kept has higher confidence
                if item['label'] == kept['label'] or kept['score'] > item['score']:
                    is_contained = True
                    break
        if not is_contained:
            keep.append(item)
    
    return keep

def postprocess_predictions(
    predictions: List[Dict[str, Any]],
    page_height: int,
    page_width: int,
    confidence_threshold: float = 0.30,
    nms_threshold: float = 0.45
) -> List[Dict[str, Any]]:
    """
    Full Stage 4 Postprocessing Pipeline.
    1. Clips all boxes to page bounds (hard constraint).
    2. Filters low-confidence and invalid-area predictions.
    3. Applies containment suppression.
    4. Performs same-class NMS.
    5. Performs cross-class NMS for high-overlap boxes.
    """
    page_area = page_height * page_width
    min_box_area = max(200, int(page_area * 0.0005))  # At least 0.05% of page
    
    processed = []
    for pred in predictions:
        clipped_box = clip_box_to_page_bounds(pred['box'], page_height, page_width)
        bw = clipped_box[2] - clipped_box[0]
        bh = clipped_box[3] - clipped_box[1]
        box_area = bw * bh
        
        # Filter: minimum meaningful size
        if bw >= 15 and bh >= 10 and box_area >= min_box_area:
            # Filter: not page-spanning (>85% of page)
            if box_area < page_area * 0.85:
                if pred['score'] >= confidence_threshold:
                    processed.append({
                        'box': clipped_box,
                        'label': pred['label'],
                        'score': float(pred['score'])
                    })
    
    # Containment suppression: remove smaller boxes inside larger same-class boxes
    processed = apply_containment_suppression(processed, containment_threshold=0.80)
    
    # Same-class NMS
    processed = apply_nms(processed, iou_threshold=nms_threshold, class_agnostic=False)
    
    # Cross-class NMS with stricter threshold (remove heavily overlapping different-class boxes)
    processed = apply_nms(processed, iou_threshold=0.60, class_agnostic=True)
                
    # Sort by spatial reading order (y top-to-bottom, then x left-to-right)
    processed.sort(key=lambda p: (p['box'][1], p['box'][0]))
    return processed
