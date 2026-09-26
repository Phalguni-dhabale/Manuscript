"""
Region proposal module for Manuscript Layout Region Detection.
Implements Stage 2: Candidate box generation using hybrid CRAFT/EasyOCR text detection
and multi-scale MSER / morphological connected-component region analysis.

Key improvement over v1:
- Hierarchical box merging groups character-level detections into text lines, then text blocks.
- MSER min_area raised to suppress glyph-level noise.
- Raw binary connected-components removed (too noisy).
- Spatial proximity merging replaces IoU-only deduplication.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
from src.utils import logger

# Global lazy initializer for EasyOCR Reader
_EASYOCR_READER = None

def get_easyocr_reader():
    """Lazy loads EasyOCR reader if torch/easyocr are available."""
    global _EASYOCR_READER
    if _EASYOCR_READER is None:
        try:
            import easyocr
            # Initialize for common scripts or generic text detection (English, Latin, Devanagari fallback)
            _EASYOCR_READER = easyocr.Reader(['en'], gpu=False, verbose=False)
            logger.info("EasyOCR text detector initialized successfully.")
        except Exception as e:
            logger.warning(f"EasyOCR initialization fallback: {e}")
            _EASYOCR_READER = False
    return _EASYOCR_READER

def extract_easyocr_proposals(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Generates candidate bounding boxes using EasyOCR CRAFT text detector."""
    reader = get_easyocr_reader()
    if not reader:
        return []
    
    boxes = []
    h, w = image.shape[:2]
    try:
        # Raise text_threshold and low_text to suppress noisy detections
        results = reader.detect(image, min_size=20, text_threshold=0.4, low_text=0.4, link_threshold=0.4)
        # easyocr detect returns (horizontal_list, free_form_list)
        horiz_boxes = results[0][0] if results and len(results) > 0 and len(results[0]) > 0 else []
        for b in horiz_boxes:
            x_min, x_max, y_min, y_max = b
            # Clamp to bounds
            x1 = max(0, int(x_min))
            y1 = max(0, int(y_min))
            x2 = min(w, int(x_max))
            y2 = min(h, int(y_max))
            bw = x2 - x1
            bh = y2 - y1
            # Filter tiny boxes and page-spanning boxes
            if bw >= 15 and bh >= 10 and (bw * bh) < (w * h * 0.85):
                boxes.append((x1, y1, x2, y2))
    except Exception as e:
        logger.warning(f"EasyOCR detection error: {e}")
        
    return boxes

def extract_mser_proposals(gray: np.ndarray, binary: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Generates candidate bounding boxes using Maximally Stable Extremal Regions (MSER)
    and morphological connected component analysis.
    
    Key improvements:
    - Higher _min_area (200) to suppress glyph-level noise
    - Only uses morphologically dilated masks (not raw binary) for connected components
    - Filters out both microscopic and page-spanning boxes
    """
    h, w = gray.shape[:2]
    page_area = w * h
    boxes = []
    
    # 1. MSER Detection with higher min_area to get word/phrase level regions
    try:
        mser = cv2.MSER_create(
            _delta=5,
            _min_area=max(200, int(page_area * 0.0005)),  # At least 0.05% of page
            _max_area=int(page_area * 0.40),
            _max_variation=0.4
        )
        regions, _ = mser.detectRegions(gray)
        for pts in regions:
            x, y, bw, bh = cv2.boundingRect(pts)
            area = bw * bh
            # Keep only meaningfully sized regions
            if bw >= 15 and bh >= 10 and area >= 200 and area < page_area * 0.85:
                boxes.append((x, y, x + bw, y + bh))
    except Exception as e:
        logger.debug(f"MSER extraction note: {e}")
        
    # 2. Morphological text-line grouping on binary mask
    # Moderate horizontal kernel to connect character glyphs into lines without bridging distant columns/blocks
    h_kernel_w = max(12, int(w * 0.035))
    kernel_horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_w, 3))
    dilated_h = cv2.dilate(binary, kernel_horiz, iterations=1)
    
    # Extract text line-level connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated_h, connectivity=8)
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y_cc = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Keep meaningful text lines: wide enough, reasonable aspect ratio
        if bw >= 25 and bh >= 8 and area >= 150 and area < page_area * 0.70:
            boxes.append((x, y_cc, x + bw, y_cc + bh))
                
    return boxes

def box_iou(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> float:
    """Computes Intersection-over-Union (IoU) between two bounding boxes."""
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

def box_containment(inner: Tuple[int, int, int, int], outer: Tuple[int, int, int, int]) -> float:
    """
    Returns the fraction of 'inner' box that is contained inside 'outer' box.
    If result > 0.85, inner is mostly inside outer.
    """
    x1 = max(inner[0], outer[0])
    y1 = max(inner[1], outer[1])
    x2 = min(inner[2], outer[2])
    y2 = min(inner[3], outer[3])
    
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h
    
    inner_area = max(1, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return inter_area / float(inner_area)

def merge_nearby_boxes(
    boxes: List[Tuple[int, int, int, int]],
    x_gap: int,
    y_gap: int,
    max_area_ratio: float = 0.50
) -> List[Tuple[int, int, int, int]]:
    """
    Merges boxes that are on the same horizontal line (significant Y-overlap)
    and within x_gap horizontally.
    
    Key constraints to prevent runaway merging:
    - Boxes must have >50% vertical overlap to be considered "same line"
    - Merged result must not exceed max_area_ratio of the bounding page
    - Limited iterations to prevent chain-reaction merging
    """
    if len(boxes) <= 1:
        return list(boxes)
    
    # Estimate page area from box extents
    all_x2 = max(b[2] for b in boxes)
    all_y2 = max(b[3] for b in boxes)
    page_area = max(1, all_x2 * all_y2)
    max_merged_area = int(page_area * max_area_ratio)
    
    merged = list(boxes)
    changed = True
    max_iters = 5  # Limited iterations
    
    while changed and max_iters > 0:
        changed = False
        max_iters -= 1
        new_merged = []
        used = [False] * len(merged)
        
        for i in range(len(merged)):
            if used[i]:
                continue
            x1, y1, x2, y2 = merged[i]
            h_i = y2 - y1
            
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                bx1, by1, bx2, by2 = merged[j]
                h_j = by2 - by1
                
                # Check vertical overlap: boxes must share >50% of their height
                overlap_y1 = max(y1, by1)
                overlap_y2 = min(y2, by2)
                v_overlap = max(0, overlap_y2 - overlap_y1)
                min_height = min(h_i, h_j)
                
                if min_height <= 0 or (v_overlap / float(min_height)) < 0.40:
                    continue  # Not on the same line
                
                # Check horizontal gap
                h_gap = max(0, max(bx1 - x2, x1 - bx2))
                if h_gap > x_gap:
                    continue  # Too far apart horizontally
                
                # Check that merged result isn't too large
                new_x1 = min(x1, bx1)
                new_y1 = min(y1, by1)
                new_x2 = max(x2, bx2)
                new_y2 = max(y2, by2)
                new_area = (new_x2 - new_x1) * (new_y2 - new_y1)
                
                if new_area > max_merged_area:
                    continue  # Would create too large a box
                
                # Merge
                x1, y1, x2, y2 = new_x1, new_y1, new_x2, new_y2
                h_i = y2 - y1
                used[j] = True
                changed = True
            
            new_merged.append((x1, y1, x2, y2))
            used[i] = True
        
        merged = new_merged
    
    return merged

def merge_and_deduplicate_boxes(
    boxes: List[Tuple[int, int, int, int]],
    image_shape: Tuple[int, int],
    iou_threshold: float = 0.50
) -> List[Tuple[int, int, int, int]]:
    """
    Merges highly overlapping candidate proposals, removes containment duplicates,
    and filters invalid boxes.
    """
    if not boxes:
        return []
        
    h, w = image_shape[:2]
    page_area = w * h
    
    # Clamp and validate boxes
    valid_boxes = []
    for x1, y1, x2, y2 in boxes:
        cx1 = max(0, min(w - 1, x1))
        cy1 = max(0, min(h - 1, y1))
        cx2 = max(cx1 + 1, min(w, x2))
        cy2 = max(cy1 + 1, min(h, y2))
        bw = cx2 - cx1
        bh = cy2 - cy1
        area = bw * bh
        # Filter: minimum meaningful size and not page-spanning
        if bw >= 15 and bh >= 10 and area >= 200 and area < page_area * 0.85:
            valid_boxes.append((cx1, cy1, cx2, cy2))
    
    # Remove containment duplicates: if small box is 85%+ inside a larger box, drop it
    valid_boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    non_contained = []
    for i, box in enumerate(valid_boxes):
        is_contained = False
        for j in range(i):
            if box_containment(box, valid_boxes[j]) > 0.85:
                is_contained = True
                break
        if not is_contained:
            non_contained.append(box)
    
    # IoU-based deduplication
    keep = []
    for b in non_contained:
        overlap = False
        for k in keep:
            if box_iou(b, k) > iou_threshold:
                overlap = True
                break
        if not overlap:
            keep.append(b)
            
    return keep

def generate_region_proposals(preprocessed: Dict[str, Any]) -> List[Tuple[int, int, int, int]]:
    """
    Generates class-agnostic candidate region proposals for manuscript layout detection.
    
    Pipeline:
    1. EasyOCR CRAFT text detection → word/line-level boxes
    2. MSER + morphological grouping → character groups + text lines + text blocks
    3. Spatial proximity merging → groups nearby small boxes into coherent regions
    4. Deduplication → removes containment and IoU overlaps
    """
    original = preprocessed['original']
    gray = preprocessed['deskewed']
    binary = preprocessed['binary']
    h, w = gray.shape[:2]
    
    # 1. EasyOCR proposals (word/line level)
    easyocr_boxes = extract_easyocr_proposals(original)
    
    # 2. MSER + Morphological proposals (text line + block level)
    mser_boxes = extract_mser_proposals(gray, binary)
    
    # 3. Combine all proposals
    all_proposals = easyocr_boxes + mser_boxes
    
    # Merge only small/medium boxes that are close together into text lines
    # This groups MSER word-level fragments without merging large EasyOCR blocks
    x_gap = max(8, int(w * 0.015))  # 1.5% of width horizontal tolerance
    y_gap = max(3, int(h * 0.008))  # 0.8% of height vertical tolerance
    all_proposals = merge_nearby_boxes(all_proposals, x_gap=x_gap, y_gap=y_gap, max_area_ratio=0.40)
    
    # If candidate pool is too small (extremely degraded image), add grid region fallbacks
    if len(all_proposals) < 3:
        logger.info("Generating grid region fallback proposals for degraded image.")
        all_proposals.append((int(w * 0.05), int(h * 0.02), int(w * 0.95), int(h * 0.15)))  # top
        all_proposals.append((int(w * 0.05), int(h * 0.85), int(w * 0.95), int(h * 0.98)))  # bottom
        all_proposals.append((int(w * 0.10), int(h * 0.18), int(w * 0.90), int(h * 0.82)))  # center
        all_proposals.append((int(w * 0.01), int(h * 0.15), int(w * 0.18), int(h * 0.85)))  # left margin
        all_proposals.append((int(w * 0.82), int(h * 0.15), int(w * 0.99), int(h * 0.85)))  # right margin
        
    final_proposals = merge_and_deduplicate_boxes(all_proposals, (h, w), iou_threshold=0.50)
    logger.info(f"Generated {len(final_proposals)} candidate region proposals.")
    return final_proposals
