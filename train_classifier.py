"""
Model training script for Manuscript Layout Region Detection.
Trains a 5-class Gradient Boosting Classifier on geometric features extracted from manuscript layout crops
or synthetic manuscript distributions (palm-leaf, paper, single/multi-column).
Saves trained classifier weights to models/classifier.pkl.
"""

import os
import argparse
import pickle
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from src.classifier import LayoutClassifier, CLASS_NAMES, CLASS_TO_ID, DEFAULT_MODEL_PATH
from src.utils import logger

def train_and_save_model(model_out_path: str = DEFAULT_MODEL_PATH):
    """
    Trains the layout region classifier and validates accuracy metrics.
    """
    logger.info("Initializing manuscript layout region classifier training...")
    
    # Delete existing model to force retraining
    if os.path.exists(model_out_path):
        os.remove(model_out_path)
        logger.info(f"Removed existing model at {model_out_path}")
    
    classifier = LayoutClassifier(model_path=model_out_path)
    
    # Train model (will retrain since model file was deleted)
    classifier.train_seed_model()
    
    # Validation evaluation with diverse test samples
    np.random.seed(123)
    test_X, test_y = [], []
    
    # Generate diverse validation samples per class
    for _ in range(80):
        # Header variations
        feat = classifier._generate_sample(
            rel_y_range=(0.01, 0.10), rel_h_range=(0.03, 0.08),
            rel_x_range=(0.10, 0.30), rel_w_range=(0.40, 0.80),
            density_range=(0.15, 0.55)
        )
        test_X.append(feat)
        test_y.append(CLASS_TO_ID['header'])
        
    for _ in range(80):
        # Footer variations
        feat = classifier._generate_sample(
            rel_y_range=(0.85, 0.95), rel_h_range=(0.03, 0.08),
            rel_x_range=(0.10, 0.35), rel_w_range=(0.30, 0.80),
            density_range=(0.15, 0.55)
        )
        test_X.append(feat)
        test_y.append(CLASS_TO_ID['footer'])
        
    for _ in range(80):
        # Main text variations (large blocks)
        feat = classifier._generate_sample(
            rel_y_range=(0.15, 0.30), rel_h_range=(0.25, 0.55),
            rel_x_range=(0.12, 0.25), rel_w_range=(0.45, 0.75),
            density_range=(0.25, 0.65)
        )
        test_X.append(feat)
        test_y.append(CLASS_TO_ID['main_text'])
    
    for _ in range(40):
        # Main text single lines
        feat = classifier._generate_sample(
            rel_y_range=(0.20, 0.75), rel_h_range=(0.02, 0.06),
            rel_x_range=(0.10, 0.20), rel_w_range=(0.50, 0.80),
            density_range=(0.20, 0.55)
        )
        test_X.append(feat)
        test_y.append(CLASS_TO_ID['main_text'])
        
    for _ in range(80):
        # Side text variations
        is_left = np.random.rand() > 0.5
        if is_left:
            feat = classifier._generate_sample(
                rel_y_range=(0.15, 0.50), rel_h_range=(0.10, 0.35),
                rel_x_range=(0.00, 0.06), rel_w_range=(0.05, 0.15),
                density_range=(0.10, 0.45)
            )
        else:
            feat = classifier._generate_sample(
                rel_y_range=(0.15, 0.50), rel_h_range=(0.10, 0.35),
                rel_x_range=(0.82, 0.93), rel_w_range=(0.05, 0.15),
                density_range=(0.10, 0.45)
            )
        test_X.append(feat)
        test_y.append(CLASS_TO_ID['side_text'])
        
    for _ in range(80):
        # Filler variations
        feat = classifier._generate_sample(
            rel_y_range=(0.10, 0.85), rel_h_range=(0.005, 0.04),
            rel_x_range=(0.05, 0.85), rel_w_range=(0.005, 0.06),
            density_range=(0.01, 0.20)
        )
        test_X.append(feat)
        test_y.append(CLASS_TO_ID['filler'])
            
    test_X = np.array(test_X, dtype=np.float32)
    test_y = np.array(test_y, dtype=np.int32)
    
    preds = classifier.model.predict(test_X)
    acc = accuracy_score(test_y, preds)
    logger.info(f"Validation Accuracy: {acc * 100:.2f}%")
    logger.info("\n" + classification_report(test_y, preds, target_names=CLASS_NAMES))
    
    logger.info(f"Model training complete. Saved weights to {model_out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Manuscript Layout Region Classifier")
    parser.add_argument('--output-model', type=str, default=DEFAULT_MODEL_PATH, help="Path to save output model pickle")
    args = parser.parse_args()
    train_and_save_model(args.output_model)
