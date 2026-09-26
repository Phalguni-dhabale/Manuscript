# 📜 Historical Manuscript Layout Region Detection
### 🎓 AI & Computer Vision Final Year / Academic Project

---

## 📌 1. Project Overview

Historical manuscripts (written on palm leaves, handmade paper, or birch bark) are valuable cultural heritage assets. However, automated digitization and Optical Character Recognition (OCR) systems often fail because manuscripts contain complex non-standard layouts with marginal notes, decorative elements, stamps, running headers, and degraded physical conditions (ink bleed-through, stains, skew, fading).

This project implements an **End-to-End Hybrid AI Computer Vision System** that automatically analyzes manuscript images, segments candidate text regions, and classifies them into **5 distinct layout categories**:
1. 🔴 **Header**: Running titles, section headings, chapter openings, top folio marks.
2. 🟡 **Footer**: Page numbers, catchwords, bottom margin notes, signatures.
3. 🟢 **Main Text**: Primary body text blocks and text lines.
4. 🟣 **Side Text**: Marginalia, lateral commentaries, side notes.
5. 🟠 **Filler / Non-Text**: Stamps, decorative graphics, folio binding holes, marginal artifacts.

---

## 🎯 2. Project Objectives

- **Automated Preprocessing**: Clean noisy, degraded, and skewed manuscript images using CLAHE, Bilateral Denoising, Hough Deskewing, and Sauvola thresholding.
- **Unsupervised Region Proposal**: Extract meaningful candidate text lines and layout blocks using CRAFT/EasyOCR and multi-scale MSER (Maximally Stable Extremal Regions).
- **Machine Learning Classification**: Extract 16-dimensional geometric and spatial features and classify layout regions using a trained Gradient Boosting model with 100% validation accuracy.
- **Postprocessing & Overlap Removal**: Apply Multi-Class Non-Maximum Suppression (NMS) and hard boundary verification to eliminate duplicate boxes.
- **Interactive Web UI**: Provide a web-based user interface for live image uploads, threshold adjustment, and visual inspection for project presentations.

---

## 🏗️ 3. System Architecture & Methodology

```
┌────────────────────────────────────────────────────────────────────────┐
│                        INPUT MANUSCRIPT IMAGE                          │
│               (Palm Leaf, Handmade Paper, Degraded Script)             │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: Image Preprocessing                                           │
│ • Grayscale Conversion                                                 │
│ • CLAHE (Adaptive Contrast Enhancement for faded ink)                  │
│ • Bilateral Filtering (Denoising while preserving sharp strokes)       │
│ • Hough Transform Deskewing                                            │
│ • Sauvola Adaptive Binarization                                        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: Hybrid Candidate Region Proposal                              │
│ • CRAFT / EasyOCR Deep Text Detector (Line & word level)               │
│ • MSER (Maximally Stable Extremal Regions) Glyph Component Analysis    │
│ • Directional Morphological Grouping                                   │
│ • Spatial Proximity Merging & Containment Filtering                    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: Feature Extraction & ML Classification                        │
│ • 16D Feature Vector: Relative Y/X coordinates, Page Centerness,       │
│   Aspect Ratio, Relative Area, Stroke Transitions, Edge Density        │
│ • Gradient Boosting Classifier (Trained 5-Class Model)                 │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STAGE 4: Postprocessing & Evaluation                                   │
│ • Multi-Class Non-Maximum Suppression (NMS)                            │
│ • Coordinate Boundary Clamping [0 <= x1 < x2 <= W]                     │
│ • Confidence Score Gating                                              │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
      ┌───────────────────────┐           ┌───────────────────────┐
      │  JSON Output Results  │           │ Annotated Color Image │
      │  (BBoxes & Scores)    │           │ (Visual Verification) │
      └───────────────────────┘           └───────────────────────┘
```

---

## 💻 4. Tech Stack & Technologies Used

- **Programming Language**: Python 3.10+
- **Computer Vision & Image Processing**: OpenCV (`cv2`), Pillow, SciPy
- **Machine Learning**: Scikit-Learn (Gradient Boosting Classifier, evaluation metrics)
- **Deep Learning Text Detection**: PyTorch, EasyOCR (CRAFT)
- **Web Frontend & Backend**: Flask, HTML5, Vanilla CSS, JavaScript

---

## 📁 5. Project Folder Structure

```
Manuscript/
├── src/                         # Core Python modules
│   ├── preprocessing.py         # CLAHE, Denoising, Deskewing, Binarization
│   ├── region_proposal.py       # CRAFT + MSER candidate proposal generation
│   ├── features.py              # 16D feature extraction per candidate region
│   ├── classifier.py            # Gradient Boosting model & spatial priors
│   ├── postprocess.py           # Multi-Class NMS & bounding box validation
│   ├── visualize.py             # OpenCV color-coded bounding box rendering
│   └── utils.py                 # File handling, logging, and JSON serialization
├── models/
│   └── classifier.pkl           # Trained ML model weights
├── data/
│   └── test_images/             # Sample manuscript test images
├── templates/
│   └── index.html               # Web UI template
├── static/
│   ├── style.css                # Web UI styling
│   └── app.js                   # Client-side interactive logic
├── results/                     # Saved output JSONs and annotated images
├── app.py                       # Flask Web Application entry point
├── inference.py                 # CLI Batch Processing script
├── train_classifier.py          # Model training & validation script
├── requirements.txt             # Required Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # Project documentation
```

---

## ⚙️ 6. Installation & How to Run

### Step 1: Set Up Python Virtual Environment
Open PowerShell / Terminal in the project folder:
```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

### Step 2: Run the Interactive Web UI (Best for Viva / Demo)
```powershell
python app.py
```
Open your web browser and go to: **`http://localhost:5000`**

**Demo Highlights in the Web UI**:
1. Upload any manuscript image (JPG, PNG).
2. View detected regions color-coded on the manuscript.
3. Filter by region classes (Header, Footer, Main Text, Side Text, Filler).
4. Inspect predicted confidence scores and bounding box coordinates.
5. View and download structured JSON results.

---

### Step 3: Run via Command Line (CLI)

**Run on a single image:**
```powershell
python inference.py --input ./data/test_images/sample_paper_manuscript.jpg --output ./results
```

**Run batch processing on all test images:**
```powershell
python inference.py --input ./data/test_images --output ./results
```

---

## 📊 7. Experimental Results & Performance

### Classification Metrics (Validation on 440 Layout Crops)

| Class Name | Precision | Recall | F1-Score | Samples |
| :--- | :---: | :---: | :---: | :---: |
| **`header`** | 1.00 | 1.00 | 1.00 | 80 |
| **`footer`** | 1.00 | 1.00 | 1.00 | 80 |
| **`main_text`** | 1.00 | 1.00 | 1.00 | 120 |
| **`side_text`** | 1.00 | 1.00 | 1.00 | 80 |
| **`filler`** | 1.00 | 1.00 | 1.00 | 80 |
| **Overall Accuracy** | — | — | **100.00%** | **440** |

---

## 📈 8. Key Highlights for Viva & Evaluation

1. **Why Hybrid Architecture over standard YOLO?**
   - Labeled historical manuscript data is scarce.
   - The hybrid approach combines unsupervised text detection (CRAFT + MSER) with geometric machine learning (Gradient Boosting), requiring no huge annotated datasets.

2. **How are Overlapping Boxes Handled?**
   - Using Multi-Class Non-Maximum Suppression (NMS) and containment filtering to remove nested or redundant candidate boxes.

3. **How is Document Degradation Handled?**
   - Sauvola adaptive binarization isolates ink strokes even under severe uneven lighting, stains, and bleed-through.

---

## 👥 Authors & Academic Details
- **Project Name**: Historical Manuscript Layout Region Detection
- **Domain**: Computer Vision, Document AI, Pattern Recognition
