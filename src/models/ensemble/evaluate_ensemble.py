"""
Script for evaluating the fraud detection ensemble model.
"""
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
def load_data(data_path: str):
    """Load data from parquet file."""
    import pandas as pd
    return pd.read_parquet(data_path)

def preprocess_data(data_path: str, val_split: float = 0.0, test_split: float = 0.2, random_state: int = 42):
    """
    Load and preprocess data.
    
    Args:
        data_path: Path to the data file
        val_split: Fraction of data to use for validation (0.0 to disable validation split)
        test_split: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    
    # Load data
    logger.info(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    
    # Separate features and target
    X = df.drop(columns=['is_fraud'])
    y = df['is_fraud'].values
    
    # Convert all columns to numeric, coercing errors to NaN
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Drop any remaining non-numeric columns
    X = X.select_dtypes(include=[np.number])
    logger.info(f"Using {len(X.columns)} numeric features after preprocessing")
    
    # Convert to numpy array
    X = X.values
    
    # If no test split is needed
    if test_split <= 0:
        X_train, X_test, y_train, y_test = X, np.array([]), y, np.array([])
    else:
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_split, random_state=random_state, stratify=y
        )
    
    # If validation split is needed
    if val_split > 0 and len(X_train) > 0:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_split, random_state=random_state, stratify=y_train
        )
    else:
        X_val, y_val = np.array([]), np.array([])
    
    # Scale features
    scaler = StandardScaler()
    if len(X_train) > 0:
        X_train = scaler.fit_transform(X_train)
    if len(X_val) > 0:
        X_val = scaler.transform(X_val)
    if len(X_test) > 0:
        X_test = scaler.transform(X_test)
    else:
        X_test = np.array([])
    
    logger.info(f"Data shapes - X_train: {X_train.shape if len(X_train) > 0 else 'N/A'}, "
                f"X_val: {X_val.shape if len(X_val) > 0 else 'N/A'}, "
                f"X_test: {X_test.shape if len(X_test) > 0 else 'N/A'}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test
from src.utils.logging_config import setup_logging

# Set up logging
logger = setup_logging(__name__)

def evaluate_ensemble(
    ensemble_model_path: str,
    data_path: str,
    output_dir: str,
    threshold: float = None,
    test_split: float = 0.2,
    random_state: int = 42
):
    """
    Evaluate the ensemble model on test data.
    
    Args:
        ensemble_model_path: Path to the saved ensemble model
        data_path: Path to the test data
        output_dir: Directory to save evaluation results
        threshold: Decision threshold (if None, use model's default)
        test_split: Fraction of data to use for testing
        random_state: Random seed for reproducibility
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load ensemble model
    logger.info(f"Loading ensemble model from {ensemble_model_path}")
    ensemble = FraudDetectionEnsemble.load(ensemble_model_path)
    
    # Override threshold if specified
    if threshold is not None:
        ensemble.threshold = threshold
    
    # Load and preprocess data
    logger.info("Loading and preprocessing test data...")
    X_train, X_test, y_train, y_test, _, _ = preprocess_data(
        data_path, 
        val_split=0.0,
        test_split=test_split,
        random_state=random_state
    )
    
    # Make predictions
    logger.info("Making predictions on test data...")
    y_pred_proba = ensemble.predict_proba(X_test)
    y_pred = (y_pred_proba >= ensemble.threshold).astype(int)
    
    # Calculate metrics
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, confusion_matrix,
        precision_recall_curve, roc_curve, auc
    )
    
    # Basic metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'average_precision': average_precision_score(y_test, y_pred_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'threshold': float(ensemble.threshold),
        'classification_weight': float(ensemble.classification_weight),
        'autoencoder_weight': float(ensemble.autoencoder_weight)
    }
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Calculate precision-recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'evaluation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")
    
    # Save ROC and PR curves
    curves = {
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'roc_auc': float(roc_auc),
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'average_precision': float(metrics['average_precision'])
    }
    
    curves_path = os.path.join(output_dir, 'evaluation_curves.json')
    with open(curves_path, 'w') as f:
        json.dump(curves, f, indent=2)
    logger.info(f"Saved curves to {curves_path}")
    
    # Log metrics
    logger.info("\nTest Set Metrics:")
    for metric, value in metrics.items():
        if metric != 'confusion_matrix':
            logger.info(f"{metric}: {value:.4f}")
    
    return metrics, curves

def main():
    parser = argparse.ArgumentParser(description='Evaluate fraud detection ensemble model')
    parser.add_argument('--ensemble-model', type=str, required=True,
                        help='Path to the saved ensemble model directory')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to test data')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save evaluation results')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Decision threshold (default: use model\'s threshold)')
    parser.add_argument('--test-split', type=float, default=0.2,
                        help='Fraction of data to use for testing (default: 0.2)')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Evaluate ensemble model
    metrics, curves = evaluate_ensemble(
        ensemble_model_path=args.ensemble_model,
        data_path=args.data_path,
        output_dir=args.output_dir,
        threshold=args.threshold,
        test_split=args.test_split,
        random_state=args.random_state
    )
    
    logger.info("Ensemble model evaluation completed successfully!")

if __name__ == "__main__":
    main()
