"""
Script for training the fraud detection ensemble model.
"""
print(f"Executing script: {__name__}")
import os
import sys
import argparse
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.models.ensemble.ensemble_model import FraudDetectionEnsemble
from src.models.train_model import load_data, preprocess_data
from src.utils.logging_config import setup_logging

# Set up logging
logger = setup_logging(__name__)

def load_models(classification_model_path: str, autoencoder_model_path: str):
    """Load pre-trained classification and autoencoder models."""
    logger.info("Loading pre-trained models...")
    
    # Load classification model
    try:
        classification_model = tf.keras.models.load_model(
            classification_model_path,
            compile=False
        )
        logger.info(f"Loaded classification model from {classification_model_path}")
    except Exception as e:
        logger.warning(f"Failed to load classification model: {e}")
        classification_model = None
    
    # Load autoencoder model
    try:
        autoencoder_model = tf.keras.models.load_model(
            autoencoder_model_path,
            compile=False
        )
        logger.info(f"Loaded autoencoder model from {autoencoder_model_path}")
    except Exception as e:
        logger.warning(f"Failed to load autoencoder model: {e}")
        autoencoder_model = None
    
    return classification_model, autoencoder_model

def train_ensemble(
    data_path: str,
    classification_model_path: str,
    autoencoder_model_path: str,
    output_dir: str,
    classification_weight: float = 0.7,
    threshold: float = 0.5,
    val_split: float = 0.2,
    random_state: int = 42
):
    """
    Train the ensemble model.
    
    Args:
        data_path: Path to the training data
        classification_model_path: Path to the pre-trained classification model
        autoencoder_model_path: Path to the pre-trained autoencoder model
        output_dir: Directory to save the ensemble model
        classification_weight: Weight for the classification model (autoencoder weight = 1 - classification_weight)
        threshold: Decision threshold for the ensemble model
        val_split: Fraction of data to use for validation
        random_state: Random seed for reproducibility
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    X_train, X_val, y_train, y_val, _, _ = preprocess_data(
        data_path, 
        val_split=val_split, 
        test_split=0.0,
        random_state=random_state
    )
    
    # Load pre-trained models
    classification_model, autoencoder_model = load_models(
        classification_model_path,
        autoencoder_model_path
    )
    
    # Create and initialize ensemble model
    ensemble = FraudDetectionEnsemble(
        classification_model=classification_model,
        autoencoder_model=autoencoder_model,
        threshold=threshold,
        classification_weight=classification_weight,
        autoencoder_weight=1.0 - classification_weight
    )
    
    # Fit reconstruction error scaler on validation data
    if autoencoder_model is not None:
        logger.info("Fitting reconstruction error scaler...")
        ensemble.fit_reconstruction_scaler(X_val)
    
    # Evaluate on validation set
    logger.info("Evaluating ensemble on validation set...")
    y_pred_proba = ensemble.predict_proba(X_val)
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate metrics
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, confusion_matrix
    )
    
    metrics = {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred, zero_division=0),
        'recall': recall_score(y_val, y_pred, zero_division=0),
        'f1_score': f1_score(y_val, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_val, y_pred_proba),
        'average_precision': average_precision_score(y_val, y_pred_proba),
        'confusion_matrix': confusion_matrix(y_val, y_pred).tolist(),
        'threshold': threshold,
        'classification_weight': classification_weight,
        'autoencoder_weight': 1.0 - classification_weight
    }
    
    # Log metrics
    logger.info("\nEnsemble Model Metrics:")
    for metric, value in metrics.items():
        if metric != 'confusion_matrix':
            logger.info(f"{metric}: {value:.4f}")
    
    # Save model
    logger.info(f"Saving ensemble model to {output_dir}")
    ensemble.save(output_dir)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'ensemble_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")
    
    return ensemble, metrics

def main():
    parser = argparse.ArgumentParser(description='Train fraud detection ensemble model')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to training data')
    parser.add_argument('--classification-model', type=str, required=True,
                        help='Path to pre-trained classification model')
    parser.add_argument('--autoencoder-model', type=str, required=True,
                        help='Path to pre-trained autoencoder model')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save the ensemble model')
    parser.add_argument('--classification-weight', type=float, default=0.7,
                        help='Weight for classification model (default: 0.7)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Decision threshold (default: 0.5)')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Fraction of data to use for validation (default: 0.2)')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Train ensemble model
    ensemble, metrics = train_ensemble(
        data_path=args.data_path,
        classification_model_path=args.classification_model,
        autoencoder_model_path=args.autoencoder_model,
        output_dir=args.output_dir,
        classification_weight=args.classification_weight,
        threshold=args.threshold,
        val_split=args.val_split,
        random_state=args.random_state
    )
    
    logger.info("Ensemble model training completed successfully!")

if __name__ == "__main__":
    main()
