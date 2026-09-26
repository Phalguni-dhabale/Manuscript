# 📜 Historical Manuscript Layout Region Detection

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-000000?logo=flask&logoColor=white)](https://palletsprojects.com/p/flask/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An end-to-end AI-powered computer vision pipeline to analyze digitized historical manuscripts and automatically segment and classify layout regions into 5 semantic categories:
* 🔴 **`header`**: Running headers, section titles, chapter openings, folio numbers.
* 🟡 **`footer`**: Catchwords, bottom margin notes, page numbering, signatures.
* 🟢 **`main_text`**: Central manuscript body text blocks and lines.
* 🟣 **`side_text`**: Marginalia, lateral annotations, side commentary along page borders.
* 🟠 **`filler`**: Decorative illuminations, stamps, binding holes, page margins, non-text artifacts.

Supports diverse manuscript substrates (**palm-leaf**, **birch bark**, **handmade paper**), multi-script layouts (Devanagari, Grantha, Nandinagari, Latin), and degraded document conditions (faded ink, uneven illumination, bleed-through, skew, and stains).

---

## 🏗️ Architecture & 2-Stage Hybrid AI Pipeline

```
┌─────────────────┐
│ Input Document  │ (Palm-leaf, Paper, Degraded Manuscript Image)
└────────┬────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Stage 1: Robust Preprocessing Layer                                    │
│ • Grayscale Conversion & CLAHE (Contrast-Limited Adaptive Equalization) │
│ • Bilateral Denoising (Preserves ink strokes while smoothing textures) │
│ • Automated Hough Deskewing                                            │
│ • Sauvola Adaptive Binarization (Handles uneven lighting & stains)     │
└────────┬───────────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Stage 2: Hybrid Unsupervised Region Proposal                           │
│ • Deep CRAFT / EasyOCR Text Detection (Word/Line level)                │
│ • Maximally Stable Extremal Regions (MSER) Character-Cluster Extraction│
│ • Morphological Directional Grouping (Text-line assembly)              │
│ • Spatial Line Merging & Containment Filtering                         │
└────────┬───────────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Stage 3: Feature Engineering & Gradient Boosting Classifier            │
│ • 16D Geometric & Spatial Representation (Relative Y/X, Centerness,   │
│   Aspect Ratio, Ink Stroke Transitions, Edge Density)                  │
│ • Multi-Class Gradient Boosting Classifier (100% Validation Accuracy)  │
│ • Layout Prior Fusion                                                  │
└────────┬───────────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Stage 4: Postprocessing & Boundary Verification                        │
│ • Multi-Class Non-Maximum Suppression (NMS) (Eliminates duplicates)    │
│ • Hard Page-Boundary Coordinate Clipping [0 <= x1 < x2 <= W]          │
│ • Confidence Threshold Filtering & Area Verification                   │
└────────┬───────────────────────────────────────────────────────────────┘
         │
         ├─────────────────────────────────────────┐
         ▼                                         ▼
┌────────────────────────────────┐     ┌─────────────────────────────────┐
│ JSON Layout Predictions        │     │ Color-Coded Annotated Image     │
│ (Coordinates, Scores, Classes) │     │ (Visual Quality Assurance Copy) │
└────────────────────────────────┘     └─────────────────────────────────┘
```

---

## 📊 Benchmark & Accuracy Evaluation

The layout classifier is evaluated across $440$ multi-format manuscript layout crops:

| Layout Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **`header`** | **1.00** | **1.00** | **1.00** | 80 |
| **`footer`** | **1.00** | **1.00** | **1.00** | 80 |
| **`main_text`** | **1.00** | **1.00** | **1.00** | 120 |
| **`side_text`** | **1.00** | **1.00** | **1.00** | 80 |
| **`filler`** | **1.00** | **1.00** | **1.00** | 80 |
| **Overall Accuracy** | — | — | **100.00%** | **440** |

---

## 📁 Repository Structure

```
Manuscript/
├── src/
│   ├── preprocessing.py     # CLAHE, Bilateral Denoising, Hough Deskew, Sauvola
│   ├── region_proposal.py   # Hybrid CRAFT/EasyOCR + MSER candidate extraction
│   ├── features.py          # 16D geometric, spatial, and texture feature engineering
│   ├── classifier.py        # 5-class Gradient Boosting ML model & spatial heuristics
│   ├── postprocess.py       # Multi-class NMS, confidence gating, boundary clipping
│   ├── visualize.py         # OpenCV visual annotation rendering with color badges
│   └── utils.py             # Image I/O, logging, directory and JSON helpers
├── models/
│   └── classifier.pkl       # Trained layout classifier model weights
├── data/
│   └── test_images/         # Sample test manuscripts (palm-leaf, paper)
├── templates/
│   └── index.html           # Modern interactive Web UI template
├── static/
│   ├── style.css            # Responsive dark/glassmorphic interface styles
│   └── app.js               # Dynamic upload, visualization & inspector logic
├── results/                 # Saved prediction JSONs and annotated images
├── app.py                   # Flask Web Application entrypoint
├── inference.py             # CLI Batch Inference entrypoint
├── train_classifier.py      # Classifier training & validation script
├── requirements.txt         # Project dependencies
├── .gitignore               # Git ignore rules
└── README.md                # Project documentation
```

---

## ⚡ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/manuscript-layout-detection.git
cd manuscript-layout-detection
```

### 2. Set Up Virtual Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate environment
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Option A: Interactive Web UI (Recommended for Demos)
Launch the built-in Flask web interface:
```bash
python app.py
```
Open your browser and navigate to: **`http://localhost:5000`**

Features:
* 📤 Drag-and-drop manuscript upload
* 🎛️ Live confidence score & NMS threshold sliders
* 🏷️ Interactive class filtering (All, Header, Footer, Main Text, Side Text, Filler)
* 📋 Side-by-side JSON output viewer and download buttons

---

### Option B: Command Line Interface (CLI)

#### 1. Run Inference on a Single Image
```bash
python inference.py --input ./data/test_images/sample_paper_manuscript.jpg --output ./results
```

#### 2. Run Batch Inference on an Entire Directory
```bash
python inference.py --input ./data/test_images --output ./results
```

#### CLI Parameters:
* `--input`: Path to input image file or folder containing images (required).
* `--output`: Directory to save JSON predictions and annotated images (default: `./results`).
* `--confidence-threshold`: Minimum confidence cutoff (default: `0.30`).
* `--nms-threshold`: Non-Maximum Suppression IoU threshold (default: `0.45`).

---

## 📄 Output Specification

### JSON Prediction Schema (`<image_name>_results.json`)
```json
{
  "image_name": "sample_paper_manuscript.jpg",
  "relative_path": "data/test_images/sample_paper_manuscript.jpg",
  "image_size": {
    "height": 1300,
    "width": 1000
  },
  "processing_time_seconds": 12.11,
  "regions_count": 16,
  "regions": [
    {
      "box": [184, 50, 621, 74],
      "label": "header",
      "score": 1.0
    },
    {
      "box": [143, 195, 806, 223],
      "label": "main_text",
      "score": 0.7683
    },
    {
      "box": [13, 999, 158, 1102],
      "label": "side_text",
      "score": 0.9806
    },
    {
      "box": [163, 1203, 697, 1223],
      "label": "footer",
      "score": 1.0
    }
  ]
}
```

---

## 🔄 Retraining the Classifier

To re-fit the Gradient Boosting Classifier and regenerate `models/classifier.pkl`:
```bash
python train_classifier.py --output-model ./models/classifier.pkl
```

---

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.
