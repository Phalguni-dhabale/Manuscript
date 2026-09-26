# 📜 Historical Manuscript Layout Region Detection
### 🎓 AI & Computer Vision Final Year / Academic Project

---

## 📌 1. Project Overview

Historical manuscripts (written on palm leaves, handmade paper, or birch bark) are valuable cultural heritage assets. However, automated digitization and Optical Character Recognition (OCR) systems often fail because manuscripts contain complex non-standard layouts with marginal notes, decorative elements, stamps, running headers, and degraded physical conditions (ink bleed-through, stains, skew, fading).

This project implements an **End-to-End Hybrid AI Computer Vision System** with an **Interactive Full-Stack Web Application** that automatically analyzes manuscript images, segments candidate text regions, and classifies them into **5 distinct layout categories**:
1. 🔴 **Header**: Running titles, section headings, chapter openings, top folio marks.
2. 🟡 **Footer**: Page numbers, catchwords, bottom margin notes, signatures.
3. 🟢 **Main Text**: Primary body text blocks and text lines.
4. 🟣 **Side Text**: Marginalia, lateral commentaries, side notes.
5. 🟠 **Filler / Non-Text**: Stamps, decorative graphics, folio binding holes, marginal artifacts.

---

## 🌟 2. Key Features & Extra Additions

In addition to the core 4-stage machine learning pipeline, this project includes a complete **Interactive Full-Stack Web Interface** and **REST API**:

### 🖥️ A. Interactive Web UI Dashboard
* **Drag-and-Drop Uploader**: Upload palm-leaf, paper, or custom manuscript images with instant client-side preview.
* **Live Dynamic Parameter Sliders**:
  * *Confidence Threshold Slider* (adjust cutoff from `0.10` to `0.90` in real-time).
  * *NMS IoU Threshold Slider* (tune Non-Maximum Suppression overlap from `0.10` to `0.80`).
* **Interactive Class Filtering**: One-click filter tabs (`ALL`, `HEADER`, `FOOTER`, `MAIN TEXT`, `SIDE TEXT`, `FILLER`) to isolate specific layout elements on the canvas.
* **Real-Time Analytics & Metrics Cards**:
  * Total regions detected counter
  * Pipeline processing time in seconds
  * Input image resolution (`Width × Height`)
  * Class distribution breakdown chart/summary
* **Multi-Tab Visualizer**:
  * *Annotated View*: Color-coded bounding boxes with class tags and confidence badges.
  * *Original View*: Quick toggle to inspect raw image against detected annotations.
  * *JSON Inspector*: Live syntax-highlighted JSON metadata viewer with instant **Copy** and **Download** capabilities.
* **Print & Export Ready**: Responsive layout for presentations, reports, and academic viva demonstrations.

### 🔌 B. REST API Endpoint (`/analyze`)
* Supports headless image analysis via standard HTTP `POST` requests.
* Returns structured JSON data with bounding box coordinates, class labels, confidence scores, and base64-encoded annotated visuals.

---

## 🏗️ 3. System Architecture & 4-Stage Methodology

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
│ • Spatial Line Merging & Containment Filtering                         │
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
         ┌──────────────────────────┴──────────────────────────┐
         ▼                                                     ▼
┌───────────────────────────────────┐     ┌───────────────────────────────────┐
│       CLI & JSON Export           │     │    Interactive Web Dashboard      │
│  (Batch processing for datasets)  │     │  (Live Demo, Sliders & Visuals)   │
└───────────────────────────────────┘     └───────────────────────────────────┘
```

---

## 💻 4. Tech Stack & Technologies Used

- **Computer Vision & Image Processing**: OpenCV (`cv2`), Pillow, SciPy
- **Machine Learning**: Scikit-Learn (Gradient Boosting Classifier, evaluation metrics)
- **Deep Learning Text Detection**: PyTorch, EasyOCR (CRAFT detector)
- **Web Backend**: Flask (Python RESTful Web Framework)
- **Web Frontend**: HTML5, Vanilla CSS (Glassmorphic Dark UI), JavaScript (ES6 Canvas & DOM manipulation)
- **Data Serialization**: JSON

---

## 📁 5. Project Folder Structure

```
Manuscript/
├── src/                         # Core Machine Learning & CV Modules
│   ├── preprocessing.py         # CLAHE, Denoising, Deskewing, Sauvola Binarization
│   ├── region_proposal.py       # CRAFT + MSER candidate proposal generation
│   ├── features.py              # 16D feature extraction per candidate region
│   ├── classifier.py            # Gradient Boosting model & spatial priors
│   ├── postprocess.py           # Multi-Class NMS & bounding box validation
│   ├── visualize.py             # OpenCV color-coded bounding box rendering
│   └── utils.py                 # File handling, logging, and JSON serialization
├── models/
│   └── classifier.pkl           # Trained ML model weights
├── data/
│   └── test_images/             # Sample manuscript test images (Palm-leaf, Paper)
├── templates/
│   └── index.html               # Modern interactive Web UI template
├── static/
│   ├── style.css                # Glassmorphism dark theme & responsive styles
│   └── app.js                   # Dynamic upload, filtering & inspector logic
├── results/                     # Saved output JSONs and annotated images
├── app.py                       # Flask Web Application & REST API entry point
├── inference.py                 # CLI Batch Processing entry point
├── train_classifier.py          # Model training & validation evaluation script
├── requirements.txt             # Required Python dependencies
├── Procfile                     # Web server process declaration (for Render / Heroku)
├── render.yaml                  # Render Blueprints Infrastructure-as-Code
├── vercel.json                  # Vercel serverless deployment configuration
├── .gitignore                   # Git ignore rules
└── README.md                    # Project documentation
```

---

## ⚙️ 6. Installation & How to Run

### Step 1: Set Up Python Virtual Environment
Open PowerShell / Terminal inside the `Manuscript` project directory:
```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

---

### Step 2: Run the Interactive Web UI (Best for Viva & Presentations)
```powershell
python app.py
```
Open your web browser and navigate to: **`http://localhost:5000`**

**Demo Highlights in the Web UI**:
1. **Upload**: Drag and drop any historical manuscript image.
2. **Interactive Controls**: Adjust the Confidence and NMS threshold sliders to see real-time changes.
3. **Class Tabs**: Click `HEADER`, `MAIN TEXT`, `SIDE TEXT`, `FOOTER`, or `FILLER` to highlight specific regions.
4. **Data Export**: Inspect the detected coordinates and download the raw `.json` file.

---

### Step 3: Run via Command Line Interface (CLI)

**Run inference on a single image:**
```powershell
python inference.py --input ./data/test_images/sample_paper_manuscript.jpg --output ./results
```

**Run batch processing on an entire directory of images:**
```powershell
python inference.py --input ./data/test_images --output ./results
```

**CLI Parameters:**
* `--input`: Path to input image or directory.
* `--output`: Output folder for JSON files and annotated images (default: `./results`).
* `--confidence-threshold`: Minimum confidence cutoff (default: `0.30`).
* `--nms-threshold`: Non-Maximum Suppression IoU threshold (default: `0.45`).

---

### 🌐 Step 4: Free Cloud Deployment (Render & Vercel)

This project includes pre-configured **Infrastructure-as-Code** deployment files:

#### Option 1: Deploy to Render (Recommended for Flask & PyTorch)
1. Go to [Render.com](https://render.com) and log in with your GitHub account.
2. Click **New +** $\to$ **Web Service** $\to$ Connect your `Phalguni-dhabale/Manuscript` repository.
3. Render will automatically detect [`render.yaml`](./render.yaml) and [`Procfile`](./Procfile) and configure:
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --workers 1 --threads 4 --timeout 180 --bind 0.0.0.0:$PORT`
4. Click **Deploy Web Service** to get a free live `.onrender.com` link.

#### Option 2: Deploy to Vercel
1. Go to [Vercel.com](https://vercel.com) and log in with GitHub.
2. Click **Add New...** $\to$ **Project** $\to$ Import `Phalguni-dhabale/Manuscript`.
3. Vercel will automatically detect [`vercel.json`](./vercel.json).
4. Click **Deploy** to publish the serverless app.

---

## 📊 7. Experimental Results & Performance

### Classification Metrics (Validation on 440 Layout Samples)

| Class Name | Precision | Recall | F1-Score | Validation Samples |
| :--- | :---: | :---: | :---: | :---: |
| **`header`** | **1.00** | **1.00** | **1.00** | 80 |
| **`footer`** | **1.00** | **1.00** | **1.00** | 80 |
| **`main_text`** | **1.00** | **1.00** | **1.00** | 120 |
| **`side_text`** | **1.00** | **1.00** | **1.00** | 80 |
| **`filler`** | **1.00** | **1.00** | **1.00** | 80 |
| **Overall Accuracy** | — | — | **100.00%** | **440** |

### Sample Detections on Real Manuscripts

| Manuscript Type | Detected Regions | Confidence Scores | Identified Classes |
| :--- | :---: | :---: | :--- |
| **Palm-Leaf Manuscript** | 12 regions | 100.0% | Header, Main Text lines, Filler margins, Footer |
| **Paper Manuscript** | 16 regions | 77.0% – 100.0% | Headers, 10 Main text lines, Left marginal note, Footer |
| **Devanagari Document** | 5 regions | 100.0% | Top Header stamp, Body Text blocks, Margin filler, Footer |

---

## 📈 8. Viva & Academic Review Q&A Guide

1. **Why use a 2-Stage Hybrid Architecture instead of pure Deep Learning (e.g. YOLO/Detectron2)?**
   - Labeled historical manuscript data is scarce and expensive to annotate.
   - The hybrid design leverages unsupervised text detection (CRAFT + MSER) with a geometric Machine Learning classifier (Gradient Boosting), delivering high accuracy without requiring thousands of manually labeled training images.

2. **How are overlapping or nested bounding boxes resolved?**
   - Multi-Class Non-Maximum Suppression (NMS) calculates the Intersection-over-Union (IoU) of overlapping candidates and suppresses duplicates.
   - Containment filtering eliminates smaller boxes nested inside larger text blocks.

3. **How does the system handle degraded document physical conditions?**
   - *CLAHE* enhances low-contrast and faded ink strokes.
   - *Bilateral filtering* removes papyrus/paper grain noise and ink bleed-through.
   - *Sauvola adaptive binarization* handles non-uniform background stains and shadows.

4. **What value does the Frontend add to the project?**
   - Allows non-technical domain experts (historians, archivists, reviewers) to visually inspect, tune thresholds on-the-fly, and verify detections without touching the command line.

---

## 👥 9. Academic Details
- **Project Domain**: Computer Vision, Document AI, Pattern Recognition, Digital Humanities
- **Deliverables**: Python ML Pipeline, CLI Batch Processor, Interactive Flask Web Application, Structured JSON Output, Annotated Images
