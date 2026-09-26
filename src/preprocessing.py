"""
Preprocessing module for Manuscript Layout Region Detection.
Implements Stage 1: Robustness layer (Grayscale, CLAHE, Denoise, Deskew, Adaptive Binarization).
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple

def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Converts a BGR image to 8-bit single-channel grayscale."""
    if len(image.shape) == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_clahe(gray: np.ndarray, clip_limit: float = 3.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Enhances local contrast, bringing out faded ink and balancing uneven lighting.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)

def apply_bilateral_filter(gray: np.ndarray, d: int = 7, sigma_color: float = 50.0, sigma_space: float = 50.0) -> np.ndarray:
    """
    Denoises the grayscale image using a bilateral filter.
    Smooths background noise, paper texture, stains, and bleed-through while preserving text edges.
    """
    return cv2.bilateralFilter(gray, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)

def estimate_skew_angle(gray: np.ndarray) -> float:
    """
    Estimates skew angle of manuscript image using Hough line transform
    and minimum bounding rectangle of text contours.
    Returns angle in degrees (-45 to 45).
    """
    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    
    angles = []
    if lines is not None:
        for line in lines:
            line_vec = line[0] if (hasattr(line, 'shape') and len(line.shape) > 1) else line
            if hasattr(line_vec, '__len__') and len(line_vec) == 4:
                x1, y1, x2, y2 = line_vec
                dx = int(x2) - int(x1)
                dy = int(y2) - int(y1)
                if dx == 0:
                    continue
                angle = np.degrees(np.arctan2(dy, dx))
                # Focus on near-horizontal lines typical of manuscript text lines (-30 to +30)
                if -30.0 < angle < 30.0:
                    angles.append(angle)
                
    if len(angles) >= 3:
        median_angle = float(np.median(angles))
        return median_angle
        
    # Fallback to binary contour minAreaRect angle
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) > 50:
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if -25.0 < angle < 25.0:
            return float(angle)
            
    return 0.0

def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotates image by given angle around its center, padding boundaries."""
    if abs(angle) < 0.2:
        return image.copy()
        
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate bounding box for rotated image to prevent cropping
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    border_val = 255 if len(image.shape) == 2 else (255, 255, 255)
    return cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=border_val)

def sauvola_threshold(gray: np.ndarray, window_size: int = 25, k: float = 0.2, r: float = 128.0) -> np.ndarray:
    """
    Computes Sauvola adaptive binarization for historical manuscript text detection.
    t = mean * (1 + k * (std / r - 1))
    """
    gray_f = gray.astype(np.float32)
    mean = cv2.boxFilter(gray_f, cv2.CV_32F, (window_size, window_size))
    sqr_mean = cv2.boxFilter(gray_f * gray_f, cv2.CV_32F, (window_size, window_size))
    variance = np.maximum(0, sqr_mean - mean * mean)
    std = np.sqrt(variance)
    
    threshold = mean * (1.0 + k * (std / r - 1.0))
    binary = np.zeros_like(gray, dtype=np.uint8)
    binary[gray_f < threshold] = 255  # Foreground text is white (255)
    return binary

def adaptive_binarize(gray: np.ndarray) -> np.ndarray:
    """
    Generates high-contrast binary mask emphasizing text strokes.
    Uses hybrid Sauvola + Adaptive Gaussian thresholding.
    """
    try:
        binary_sauvola = sauvola_threshold(gray, window_size=25, k=0.18)
    except Exception:
        binary_sauvola = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10
        )
    return binary_sauvola

def preprocess_image(image: np.ndarray) -> Dict[str, Any]:
    """
    Full Stage 1 Preprocessing Pipeline.
    
    Returns dictionary containing:
    - 'original': untouched original image
    - 'gray': grayscale representation
    - 'clahe': CLAHE enhanced grayscale
    - 'denoised': bilateral filtered CLAHE
    - 'skew_angle': estimated skew angle in degrees
    - 'deskewed': deskewed CLAHE image
    - 'binary': adaptive Sauvola binary mask
    """
    gray = convert_to_grayscale(image)
    clahe = apply_clahe(gray, clip_limit=2.5)
    denoised = apply_bilateral_filter(clahe, d=7, sigma_color=50, sigma_space=50)
    skew_angle = estimate_skew_angle(denoised)
    
    if abs(skew_angle) > 0.5:
        deskewed = rotate_image(denoised, skew_angle)
    else:
        deskewed = denoised.copy()
        
    binary = adaptive_binarize(deskewed)
    
    return {
        'original': image,
        'gray': gray,
        'clahe': clahe,
        'denoised': denoised,
        'skew_angle': skew_angle,
        'deskewed': deskewed,
        'binary': binary
    }
