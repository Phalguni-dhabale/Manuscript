"""
Flask Web Application for Manuscript Layout Region Detection.
Provides a browser-based UI for uploading manuscript images,
running the detection pipeline, and viewing annotated results interactively.

Usage:
  python app.py
  Then open http://localhost:5000 in your browser.
"""

import os
import sys
import time
import json
import uuid
import base64
from flask import Flask, render_template, request, jsonify, send_from_directory

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import logger, load_image, save_json_results
from src.preprocessing import preprocess_image
from src.region_proposal import generate_region_proposals
from src.classifier import LayoutClassifier
from src.postprocess import postprocess_predictions
from src.visualize import render_annotated_image

import cv2
import numpy as np

app = Flask(__name__, static_folder='static', template_folder='templates')

# Configuration
try:
    UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, 'uploads')
    RESULTS_FOLDER = os.path.join(PROJECT_ROOT, 'results')
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
except (PermissionError, OSError):
    UPLOAD_FOLDER = os.path.join('/tmp', 'uploads')
    RESULTS_FOLDER = os.path.join('/tmp', 'results')
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff', 'webp'}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max upload

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Load classifier once at startup
classifier = None

def get_classifier():
    """Lazy-load the classifier on first request."""
    global classifier
    if classifier is None:
        model_path = os.path.join(PROJECT_ROOT, 'models', 'classifier.pkl')
        classifier = LayoutClassifier(model_path=model_path)
    return classifier

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def encode_image_to_base64(image_bgr):
    """Encodes a BGR OpenCV image to base64 JPEG string."""
    _, buffer = cv2.imencode('.jpg', image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return base64.b64encode(buffer).decode('utf-8')


@app.route('/')
def index():
    """Serve the main web UI."""
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze_image():
    """
    Accepts an uploaded image, runs the full 4-stage pipeline,
    and returns JSON results + base64-encoded annotated image.
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded.'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': f'Unsupported file format. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

    # Parse optional parameters
    confidence_threshold = float(request.form.get('confidence_threshold', 0.30))
    nms_threshold = float(request.form.get('nms_threshold', 0.45))

    try:
        # Save uploaded file
        unique_id = str(uuid.uuid4())[:8]
        original_name = file.filename
        safe_name = f"{unique_id}_{original_name}"
        upload_path = os.path.join(UPLOAD_FOLDER, safe_name)
        file.save(upload_path)

        start_time = time.time()
        logger.info(f"[Web] Processing uploaded image: {original_name}")

        # Stage 1: Load image
        original_img, height, width = load_image(upload_path)

        # Stage 2: Preprocessing
        preprocessed = preprocess_image(original_img)

        # Stage 3: Region Proposal
        candidate_boxes = generate_region_proposals(preprocessed)

        # Stage 4: Classification
        clf = get_classifier()
        raw_predictions = clf.classify_boxes(
            boxes=candidate_boxes,
            page_shape=(height, width),
            binary_mask=preprocessed['binary']
        )

        # Stage 5: Postprocessing
        final_regions = postprocess_predictions(
            predictions=raw_predictions,
            page_height=height,
            page_width=width,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold
        )

        elapsed = time.time() - start_time
        logger.info(f"[Web] Detected {len(final_regions)} regions in {elapsed:.2f}s")

        # Render annotated image
        annotated_img = render_annotated_image(original_img, final_regions)

        # Encode images to base64
        original_b64 = encode_image_to_base64(original_img)
        annotated_b64 = encode_image_to_base64(annotated_img)

        # Count regions by class
        class_counts = {}
        for r in final_regions:
            label = r['label']
            class_counts[label] = class_counts.get(label, 0) + 1

        # Save JSON results to results/ folder
        basename = os.path.splitext(original_name)[0]
        json_path = os.path.join(RESULTS_FOLDER, f"{basename}_results.json")
        result_data = {
            "image_name": original_name,
            "image_size": {"height": height, "width": width},
            "processing_time_seconds": round(elapsed, 4),
            "regions_count": len(final_regions),
            "regions": final_regions
        }
        save_json_results(json_path, result_data)

        # Save annotated image
        annotated_path = os.path.join(RESULTS_FOLDER, f"{basename}_annotated.jpg")
        cv2.imwrite(annotated_path, annotated_img)

        # Build response
        response = {
            'success': True,
            'image_name': original_name,
            'image_size': {'height': height, 'width': width},
            'processing_time': round(elapsed, 2),
            'total_regions': len(final_regions),
            'class_counts': class_counts,
            'regions': final_regions,
            'original_image': original_b64,
            'annotated_image': annotated_b64
        }

        # Clean up uploaded file
        try:
            os.remove(upload_path)
        except OSError:
            pass

        return jsonify(response)

    except Exception as e:
        logger.error(f"[Web] Error processing image: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/results/<path:filename>')
def serve_result(filename):
    """Serve files from results directory."""
    return send_from_directory(RESULTS_FOLDER, filename)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("\n" + "=" * 60)
    print("  Manuscript Layout Region Detection — Web UI")
    print(f"  Open in browser: http://localhost:{port}")
    print("=" * 60 + "\n")
    app.run(host='0.0.0.0', port=port, debug=False)

