"""
Inference CLI script for Manuscript Layout Region Detection.
Implements batch execution pipeline for single images or full directories.

Usage:
  python inference.py --input ./data/test_images --output ./results
"""

import os
import sys
import argparse
import time
from typing import Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import logger, collect_image_paths, load_image, save_json_results
from src.preprocessing import preprocess_image
from src.region_proposal import generate_region_proposals
from src.classifier import LayoutClassifier
from src.postprocess import postprocess_predictions
from src.visualize import render_annotated_image, save_annotated_image

def process_single_image(
    image_path: str,
    output_dir: str,
    classifier: LayoutClassifier,
    confidence_threshold: float = 0.30,
    nms_threshold: float = 0.45
) -> Dict[str, Any]:
    """
    Processes a single manuscript image end-to-end:
    Stage 1: Preprocessing
    Stage 2: Region Proposal
    Stage 3: Region Classification
    Stage 4: Postprocessing & JSON/Image Export
    """
    start_time = time.time()
    logger.info(f"Processing image: {image_path}")
    
    # 1. Load image (original untouched)
    original_img, height, width = load_image(image_path)
    
    # 2. Stage 1 — Preprocessing
    preprocessed = preprocess_image(original_img)
    
    # 3. Stage 2 — Region Proposal
    candidate_boxes = generate_region_proposals(preprocessed)
    
    # 4. Stage 3 — Region Classification
    raw_predictions = classifier.classify_boxes(
        boxes=candidate_boxes,
        page_shape=(height, width),
        binary_mask=preprocessed['binary']
    )
    
    # 5. Stage 4 — Postprocessing (clipping to page bounds, NMS, thresholding)
    final_regions = postprocess_predictions(
        predictions=raw_predictions,
        page_height=height,
        page_width=width,
        confidence_threshold=confidence_threshold,
        nms_threshold=nms_threshold
    )
    
    elapsed = time.time() - start_time
    logger.info(f"Detected {len(final_regions)} layout regions in {elapsed:.2f}s")
    
    # Prepare JSON structure
    rel_image_path = os.path.relpath(image_path, start=os.getcwd()) if os.path.isabs(image_path) else image_path
    image_basename = os.path.splitext(os.path.basename(image_path))[0]
    
    result_data = {
        "image_name": os.path.basename(image_path),
        "relative_path": rel_image_path,
        "image_size": {
            "height": height,
            "width": width
        },
        "processing_time_seconds": round(elapsed, 4),
        "regions_count": len(final_regions),
        "regions": final_regions
    }
    
    # Save JSON metadata in output_dir
    os.makedirs(output_dir, exist_ok=True)
    json_output_path = os.path.join(output_dir, f"{image_basename}_results.json")
    save_json_results(json_output_path, result_data)
    
    # Render and save annotated visual copy
    annotated_canvas = render_annotated_image(original_img, final_regions)
    annotated_output_path = os.path.join(output_dir, f"{image_basename}_annotated.jpg")
    save_annotated_image(annotated_output_path, annotated_canvas)
    
    return result_data

def run_batch_inference(
    input_path: str,
    output_dir: str,
    confidence_threshold: float = 0.30,
    nms_threshold: float = 0.45,
    model_path: str = None
) -> None:
    """
    Executes batch inference across all images under input_path.
    """
    image_paths = collect_image_paths(input_path)
    if not image_paths:
        logger.error(f"No images found to process for input: {input_path}")
        return
        
    logger.info(f"Found {len(image_paths)} image(s) for batch processing.")
    
    # Load layout classifier
    if model_path is None:
        model_path = os.path.join(PROJECT_ROOT, 'models', 'classifier.pkl')
        
    classifier = LayoutClassifier(model_path=model_path)
    
    batch_summary = []
    for idx, img_path in enumerate(image_paths, 1):
        logger.info(f"[{idx}/{len(image_paths)}] Batch item: {os.path.basename(img_path)}")
        try:
            res = process_single_image(
                image_path=img_path,
                output_dir=output_dir,
                classifier=classifier,
                confidence_threshold=confidence_threshold,
                nms_threshold=nms_threshold
            )
            batch_summary.append(res)
        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}", exc_info=True)
            
    # Save overall summary JSON if multiple images processed
    if len(batch_summary) > 1:
        summary_json_path = os.path.join(output_dir, "batch_summary.json")
        save_json_results(summary_json_path, {"total_images": len(batch_summary), "batch_results": batch_summary})
        logger.info(f"Batch processing completed. Master summary saved to {summary_json_path}")
    else:
        logger.info(f"Processing completed. Results saved under {output_dir}")

def main():
    parser = argparse.ArgumentParser(
        description="Manuscript Layout Region Detection CLI Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help="Path to input image file or directory containing manuscript images."
    )
    parser.add_argument(
        '--output',
        type=str,
        default="./results",
        help="Path to directory where output JSON metadata and annotated images will be saved."
    )
    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.30,
        help="Minimum confidence score threshold for layout region detection."
    )
    parser.add_argument(
        '--nms-threshold',
        type=float,
        default=0.45,
        help="IoU threshold for Non-Maximum Suppression."
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help="Optional path to custom classifier pickle model weights."
    )
    
    args = parser.parse_args()
    
    run_batch_inference(
        input_path=args.input,
        output_dir=args.output,
        confidence_threshold=args.confidence_threshold,
        nms_threshold=args.nms_threshold,
        model_path=args.model_path
    )

if __name__ == '__main__':
    main()
