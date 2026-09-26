"""
Classifier module for Manuscript Layout Region Detection.
Implements Stage 3: 5-class region classification (header, footer, main_text, side_text, filler)
using an ensemble of trained Gradient Boosting / Random Forest model and heuristic geometric priors.

Key improvements over v1:
- Much broader synthetic training data covering realistic box geometries
- Main text includes single-line, multi-line, and small paragraph variants
- Stronger geometric priors with clearer decision boundaries
- Area-based sanity checks before classification
"""

import os
import pickle
import numpy as np
from typing import List, Tuple, Dict, Any
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from src.features import extract_batch_features, extract_box_features
from src.utils import logger

# Target class taxonomy
CLASS_NAMES = ['header', 'footer', 'main_text', 'side_text', 'filler']
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'classifier.pkl')

class LayoutClassifier:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model_path = os.path.abspath(model_path)
        self.model = None
        self.load_or_create_model()
        
    def load_or_create_model(self):
        """Loads trained pickle weights or builds/trains robust default classifier."""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info(f"Loaded trained classifier weights from {self.model_path}")
                return
            except Exception as e:
                logger.warning(f"Could not load model weights from {self.model_path}: {e}. Retraining default model.")
                
        # Build synthetic seed training dataset matching manuscript layout geometries
        self.train_seed_model()
        
    def _generate_sample(self, rel_y_range, rel_h_range, rel_x_range, rel_w_range, density_range):
        """Helper to generate one synthetic training sample with randomized features."""
        rel_y_min = np.random.uniform(*rel_y_range)
        rel_h = np.random.uniform(*rel_h_range)
        rel_y_max = min(1.0, rel_y_min + rel_h)
        rel_y_center = (rel_y_min + rel_y_max) / 2.0
        
        rel_x_min = np.random.uniform(*rel_x_range)
        rel_w = np.random.uniform(*rel_w_range)
        rel_x_max = min(1.0, rel_x_min + rel_w)
        rel_x_center = (rel_x_min + rel_x_max) / 2.0
        
        rel_width = rel_x_max - rel_x_min
        rel_height = rel_y_max - rel_y_min
        rel_area = rel_width * rel_height
        aspect = rel_width / max(0.001, rel_height)
        
        dist_top = rel_y_min
        dist_bottom = 1.0 - rel_y_max
        dist_left = rel_x_min
        dist_right = 1.0 - rel_x_max
        min_dist = min(dist_top, dist_bottom, dist_left, dist_right)
        density = np.random.uniform(*density_range)
        
        return [rel_y_min, rel_y_max, rel_y_center, rel_x_min, rel_x_max, rel_x_center,
                rel_width, rel_height, rel_area, aspect, dist_top, dist_bottom,
                dist_left, dist_right, min_dist, density]
        
    def train_seed_model(self):
        """
        Trains a robust seed model using synthetically generated geometric priors
        representing palm-leaf manuscripts, paper manuscripts, single/multi-column layouts,
        top running headers, bottom signatures/catchwords, marginalia side text, and decorative fillers.
        
        Key improvement: Generates much more diverse training data covering:
        - Single text lines (small height, wide width) → main_text
        - Text blocks of varying sizes → main_text
        - Narrow header/footer strips → header/footer
        - Marginal annotations → side_text
        - Small isolated artifacts → filler
        """
        np.random.seed(42)
        X_train = []
        y_train = []
        
        # ─── 1. HEADER samples ───
        # Wide horizontal strip at the top 20% of page
        for _ in range(600):
            feat = self._generate_sample(
                rel_y_range=(0.00, 0.12),
                rel_h_range=(0.02, 0.12),
                rel_x_range=(0.05, 0.35),
                rel_w_range=(0.30, 0.90),
                density_range=(0.15, 0.65)
            )
            X_train.append(feat)
            y_train.append(CLASS_TO_ID['header'])
        
        # Narrow folio numbers / running headers at top
        for _ in range(200):
            feat = self._generate_sample(
                rel_y_range=(0.00, 0.08),
                rel_h_range=(0.02, 0.06),
                rel_x_range=(0.20, 0.60),
                rel_w_range=(0.10, 0.40),
                density_range=(0.10, 0.50)
            )
            X_train.append(feat)
            y_train.append(CLASS_TO_ID['header'])
            
        # ─── 2. FOOTER samples ───
        # Wide horizontal strip at the bottom 20% of page
        for _ in range(600):
            feat = self._generate_sample(
                rel_y_range=(0.82, 0.96),
                rel_h_range=(0.02, 0.12),
                rel_x_range=(0.05, 0.40),
                rel_w_range=(0.25, 0.90),
                density_range=(0.15, 0.65)
            )
            X_train.append(feat)
            y_train.append(CLASS_TO_ID['footer'])
        
        # Narrow catchwords / page numbers at bottom
        for _ in range(200):
            feat = self._generate_sample(
                rel_y_range=(0.88, 0.98),
                rel_h_range=(0.01, 0.05),
                rel_x_range=(0.30, 0.70),
                rel_w_range=(0.08, 0.30),
                density_range=(0.10, 0.50)
            )
            X_train.append(feat)
            y_train.append(CLASS_TO_ID['footer'])
            
        # ─── 3. MAIN TEXT samples ───
        # Large central text blocks (manuscript body)
        for _ in range(800):
            feat = self._generate_sample(
                rel_y_range=(0.10, 0.30),
                rel_h_range=(0.30, 0.70),
                rel_x_range=(0.08, 0.25),
                rel_w_range=(0.50, 0.85),
                density_range=(0.25, 0.75)
            )
            X_train.append(feat)
            y_train.append(CLASS_TO_ID['main_text'])
        
        # Medium text blocks (half-page or column)
        for _ in range(500):
            feat = self._generate_sample(
                rel_y_range=(0.12, 0.45),
                rel_h_range=(0.15, 0.45),
                rel_x_range=(0.10, 0.40),
                rel_w_range=(0.25, 0.65),
                density_range=(0.20, 0.70)
            )
            X_train.append(feat)
            y_train.append(CLASS_TO_ID['main_text'])
        
        # Single text lines in central area (common in palm-leaf manuscripts)
        for _ in range(400):
            feat = self._generate_sample(
                rel_y_range=(0.15, 0.80),
                rel_h_range=(0.02, 0.08),
                rel_x_range=(0.08, 0.20),
                rel_w_range=(0.40, 0.85),
                density_range=(0.20, 0.65)
            )
            X_train.append(feat)
            y_train.append(CLASS_TO_ID['main_text'])
        
        # Small text paragraphs
        for _ in range(300):
            feat = self._generate_sample(
                rel_y_range=(0.15, 0.65),
                rel_h_range=(0.08, 0.25),
                rel_x_range=(0.12, 0.35),
                rel_w_range=(0.30, 0.70),
                density_range=(0.20, 0.60)
            )
            X_train.append(feat)
            y_train.append(CLASS_TO_ID['main_text'])
            
        # ─── 4. SIDE TEXT samples ───
        # Left margin annotations
        for _ in range(400):
            feat = self._generate_sample(
                rel_y_range=(0.10, 0.55),
                rel_h_range=(0.08, 0.45),
                rel_x_range=(0.00, 0.08),
                rel_w_range=(0.04, 0.18),
                density_range=(0.10, 0.55)
            )
            X_train.append(feat)
            y_train.append(CLASS_TO_ID['side_text'])
        
        # Right margin annotations
        for _ in range(400):
            feat = self._generate_sample(
                rel_y_range=(0.10, 0.55),
                rel_h_range=(0.08, 0.45),
                rel_x_range=(0.80, 0.95),
                rel_w_range=(0.04, 0.18),
                density_range=(0.10, 0.55)
            )
            X_train.append(feat)
            y_train.append(CLASS_TO_ID['side_text'])
            
        # ─── 5. FILLER samples ───
        # Small isolated artifacts (stamps, holes, marks)
        for _ in range(500):
            feat = self._generate_sample(
                rel_y_range=(0.05, 0.90),
                rel_h_range=(0.005, 0.05),
                rel_x_range=(0.02, 0.90),
                rel_w_range=(0.005, 0.08),
                density_range=(0.01, 0.25)
            )
            X_train.append(feat)
            y_train.append(CLASS_TO_ID['filler'])
        
        # Medium decorative elements / binding holes
        for _ in range(200):
            feat = self._generate_sample(
                rel_y_range=(0.20, 0.80),
                rel_h_range=(0.03, 0.10),
                rel_x_range=(0.00, 0.05),
                rel_w_range=(0.02, 0.06),
                density_range=(0.02, 0.20)
            )
            X_train.append(feat)
            y_train.append(CLASS_TO_ID['filler'])

        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.int32)
        
        clf = GradientBoostingClassifier(
            n_estimators=200, 
            learning_rate=0.1, 
            max_depth=5, 
            min_samples_leaf=5,
            random_state=42
        )
        clf.fit(X_train, y_train)
        self.model = clf
        
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump(clf, f)
        logger.info(f"Trained and saved seed classifier to {self.model_path}")
        
    def classify_boxes(
        self,
        boxes: List[Tuple[int, int, int, int]],
        page_shape: Tuple[int, int],
        binary_mask: np.ndarray = None
    ) -> List[Dict[str, Any]]:
        """
        Classifies candidate boxes into 5 target layout classes.
        Combines model class probabilities with geometric priors for maximum precision.
        
        Key improvements:
        - Stronger geometric priors with clearer boundaries
        - Priors applied as multiplicative boosts (not flat additions)
        - Area-based hard rules prevent misclassification of obvious regions
        """
        if not boxes:
            return []
            
        features = extract_batch_features(boxes, page_shape, binary_mask)
        probs = self.model.predict_proba(features)
        
        H, W = page_shape[:2]
        page_area = H * W
        results = []
        
        for i, box in enumerate(boxes):
            feat = features[i]
            rel_y_min = feat[0]
            rel_y_max = feat[1]
            rel_y_center = feat[2]
            rel_x_min = feat[3]
            rel_x_max = feat[4]
            rel_x_center = feat[5]
            rel_width = feat[6]
            rel_height = feat[7]
            rel_area = feat[8]
            aspect_ratio = feat[9]
            dist_top = feat[10]
            dist_bottom = feat[11]
            dist_left = feat[12]
            dist_right = feat[13]
            min_edge_dist = feat[14]
            text_density = feat[15]
            
            p_vec = probs[i].copy()
            
            # ── Geometric prior boosts ──
            
            # HEADER: top region, horizontally wide
            if rel_y_center < 0.15 and rel_width > 0.20:
                p_vec[CLASS_TO_ID['header']] *= 3.0
            elif rel_y_center < 0.20 and rel_width > 0.15:
                p_vec[CLASS_TO_ID['header']] *= 2.0
            
            # FOOTER: bottom region, horizontally wide
            if rel_y_center > 0.85 and rel_width > 0.20:
                p_vec[CLASS_TO_ID['footer']] *= 3.0
            elif rel_y_center > 0.80 and rel_width > 0.15:
                p_vec[CLASS_TO_ID['footer']] *= 2.0
                
            # MAIN TEXT: central body, significant area, good text density
            if (0.15 <= rel_y_center <= 0.85 and 
                0.20 <= rel_x_center <= 0.80 and
                rel_area > 0.03 and text_density > 0.10):
                p_vec[CLASS_TO_ID['main_text']] *= 3.5
            elif (0.12 <= rel_y_center <= 0.88 and rel_area > 0.02 and rel_width > 0.20):
                p_vec[CLASS_TO_ID['main_text']] *= 2.0
            
            # Single text lines in body area (wide and thin)
            if (0.12 <= rel_y_center <= 0.88 and rel_width > 0.30 and
                rel_height < 0.06 and text_density > 0.10):
                p_vec[CLASS_TO_ID['main_text']] *= 2.5
                
            # SIDE TEXT: narrow left or right margins, vertically extended
            if rel_x_center < 0.15 and rel_height > 0.08 and rel_width < 0.20:
                p_vec[CLASS_TO_ID['side_text']] *= 3.0
            elif rel_x_center > 0.85 and rel_height > 0.08 and rel_width < 0.20:
                p_vec[CLASS_TO_ID['side_text']] *= 3.0
                
            # FILLER: very small area, low text density
            if rel_area < 0.005:
                p_vec[CLASS_TO_ID['filler']] *= 2.5
            if text_density < 0.05:
                p_vec[CLASS_TO_ID['filler']] *= 2.0
            
            # ── Hard rule overrides for obvious cases ──
            
            # Very large central region with text → always main_text
            if (rel_area > 0.15 and 0.20 <= rel_x_center <= 0.80 and
                0.20 <= rel_y_center <= 0.80 and text_density > 0.10):
                p_vec[CLASS_TO_ID['main_text']] = max(p_vec[CLASS_TO_ID['main_text']], 0.8)
            
            # Top strip with text → header
            if rel_y_max < 0.18 and rel_width > 0.25 and text_density > 0.05:
                p_vec[CLASS_TO_ID['header']] = max(p_vec[CLASS_TO_ID['header']], 0.6)
            
            # Bottom strip with text → footer
            if rel_y_min > 0.82 and rel_width > 0.25 and text_density > 0.05:
                p_vec[CLASS_TO_ID['footer']] = max(p_vec[CLASS_TO_ID['footer']], 0.6)

            # Normalize to sum to 1.0
            p_sum = np.sum(p_vec)
            if p_sum > 0:
                p_vec = p_vec / p_sum
            else:
                p_vec = np.ones(len(CLASS_NAMES)) / len(CLASS_NAMES)
            
            pred_id = int(np.argmax(p_vec))
            confidence = float(p_vec[pred_id])
            pred_label = ID_TO_CLASS[pred_id]
            
            results.append({
                'box': [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                'label': pred_label,
                'score': round(confidence, 4)
            })
            
        return results
