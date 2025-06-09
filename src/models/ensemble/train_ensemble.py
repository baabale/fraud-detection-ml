"""
Script for training the fraud detection ensemble model.
"""
import os
import sys
import time
import argparse
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.models.ensemble.ensemble_model import FraudDetectionEnsemble
from src.models.preprocessing.data_processor import FraudDataProcessor
def load_data(data_path: str):
    """Load data from parquet file."""
    import pandas as pd
    return pd.read_parquet(data_path)

# List of 22 features expected by the classification model
CLASSIFICATION_FEATURES = [
    'amount', 'hour_of_day', 'day_of_week', 'month', 'year',
    'time_since_last_transaction', 'spending_deviation_score',
    'velocity_score', 'geo_anomaly_score', 'amount_log',
    'fraud_label', 'velocity_score_norm', 'geo_anomaly_bin',
    'device_channel', 'spending_risk', 'transaction_type',
    'merchant_category', 'location', 'payment_channel',
    'device_used', 'ip_address', 'device_hash'
]

def preprocess_data(data_path: str, val_split: float = 0.2, test_split: float = 0.0, random_state: int = 42,
                  apply_sampling: bool = False, sampling_technique: str = 'none', 
                  sampling_ratio: float = 0.5, k_neighbors: int = 5):
    """
    Load and preprocess data for the ensemble model using the unified FraudDataProcessor.
    
    Args:
        data_path (str): Path to the data file
        val_split (float): Fraction of data to use for validation
        test_split (float): Fraction of data to use for testing
        random_state (int): Random seed for reproducibility
        apply_sampling (bool): Whether to apply advanced sampling techniques
        sampling_technique (str): Sampling technique to use (none, smote, adasyn, borderline_smote)
        sampling_ratio (float): Desired ratio of minority to majority class after sampling
        k_neighbors (int): Number of nearest neighbors for sampling techniques
        
    Returns:
        tuple: X_train, X_val, X_test, y_train, y_val, y_test, data_processor
    """
    logger.info("Preprocessing data with unified data processor...")
    
    # Initialize the unified data processor
    logger.info("Initializing unified data processor for ensemble training")
    
    # We'll try both common target column names
    logger.info("Will try both 'fraud_label' and 'is_fraud' as target columns")
    target_column = 'fraud_label'  # Default first attempt
    
    data_processor = FraudDataProcessor(
        classification_features=CLASSIFICATION_FEATURES,
        autoencoder_features=None,  # Will be determined later based on autoencoder model
        target_column=target_column
    )
    
    # Process the data with error handling for different target column names
    logger.info(f"Processing data from {data_path} with unified processor")
    try:
        # First try with the default target column
        result = data_processor.process_data(
            data_path=data_path,
            val_split=val_split,
            test_split=test_split,
            apply_sampling=apply_sampling,
            sampling_technique=sampling_technique,
            sampling_ratio=sampling_ratio,
            k_neighbors=k_neighbors
        )
    except Exception as e:
        logger.warning(f"Error with target column '{target_column}': {str(e)}")
        logger.info("Trying alternative target column name 'is_fraud'")
        
        # Try with the alternative target column
        data_processor.target_column = 'is_fraud'
        try:
            result = data_processor.process_data(
                data_path=data_path,
                val_split=val_split,
                test_split=test_split,
                apply_sampling=apply_sampling,
                sampling_technique=sampling_technique,
                sampling_ratio=sampling_ratio,
                k_neighbors=k_neighbors
            )
        except Exception as e2:
            logger.error(f"Failed with both target column names: {str(e2)}")
            raise ValueError("Could not process data with either 'fraud_label' or 'is_fraud' as target column")
    
    # Log the keys available in the result for debugging
    logger.debug(f"Data processor returned keys: {list(result.keys())}")
    
    # The FraudDataProcessor returns a nested dictionary structure
    # First level keys are 'classification', 'autoencoder'
    # Second level keys are 'X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test'
    
    # Extract classification data
    if 'classification' in result:
        clf_data = result['classification']
        logger.debug(f"Classification data keys: {list(clf_data.keys())}")
        
        # Extract training data
        if 'X_train' in clf_data:
            X_train = clf_data['X_train']
            y_train = clf_data['y_train']
        else:
            raise KeyError(f"Could not find training data in classification result. Available keys: {list(clf_data.keys())}")
        
        # Extract validation data
        if val_split > 0 and 'X_val' in clf_data:
            X_val = clf_data['X_val']
            y_val = clf_data['y_val']
        else:
            X_val = None
            y_val = None
            logger.warning("No validation data found or validation split is 0")
        
        # Extract test data
        if test_split > 0 and 'X_test' in clf_data:
            X_test = clf_data['X_test']
            y_test = clf_data['y_test']
        else:
            X_test = None
            y_test = None
    else:
        raise KeyError(f"Could not find classification data in result. Available keys: {list(result.keys())}")
        
    # Store autoencoder data in the data processor for later use
    if 'autoencoder' in result:
        data_processor.autoencoder_data = result['autoencoder']
        logger.info("Stored autoencoder data in data processor for later use")
    else:
        logger.warning("No autoencoder data found in result. Will use classification data for autoencoder.")
        data_processor.autoencoder_data = None
    
    # Log the keys available in the result for debugging
    logger.debug(f"Data processor returned keys: {list(result.keys())}")
    
    # Log data shapes
    logger.info(f"Data shapes - X_train: {X_train.shape if X_train is not None and len(X_train) > 0 else 'N/A'}, "
                f"X_val: {X_val.shape if X_val is not None and len(X_val) > 0 else 'N/A'}, "
                f"X_test: {X_test.shape if X_test is not None and len(X_test) > 0 else 'N/A'}")
    
    # Ensure we have at least training data
    if X_train is None or len(X_train) == 0:
        raise ValueError("No training data available after preprocessing")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, data_processor

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

def evaluate_thresholds(y_true: np.ndarray, y_scores: np.ndarray, n_thresholds: int = 100) -> Dict[str, np.ndarray]:
    """
    Evaluate model performance at different thresholds.
    
    Args:
        y_true: True labels
        y_scores: Predicted probability scores
        n_thresholds: Number of thresholds to evaluate
        
    Returns:
        Dictionary containing metrics at each threshold
    """
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    
    thresholds = np.linspace(0, 1, n_thresholds)
    metrics = {
        'thresholds': thresholds,
        'precision': [],
        'recall': [],
        'f1': [],
        'accuracy': []
    }
    
    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        metrics['precision'].append(precision_score(y_true, y_pred, zero_division=0))
        metrics['recall'].append(recall_score(y_true, y_pred, zero_division=0))
        metrics['f1'].append(f1_score(y_true, y_pred, zero_division=0))
        metrics['accuracy'].append(accuracy_score(y_true, y_pred))
    
    # Convert to numpy arrays
    for key in ['precision', 'recall', 'f1', 'accuracy']:
        metrics[key] = np.array(metrics[key])
    
    return metrics

def find_optimal_threshold(y_true: np.ndarray, y_scores: np.ndarray, 
                          target_metric: str = 'f1', 
                          target_recall: Optional[float] = None,
                          target_precision: Optional[float] = None) -> Tuple[float, float]:
    """
    Find the optimal threshold based on different criteria.
    
    Args:
        y_true: True labels
        y_scores: Predicted probability scores
        target_metric: Metric to optimize ('f1', 'precision', 'recall', 'f2')
        target_recall: If provided, finds threshold that achieves at least this recall
        target_precision: If provided, finds threshold that achieves at least this precision
        
    Returns:
        Tuple of (optimal_threshold, metric_value)
    """
    metrics = evaluate_thresholds(y_true, y_scores)
    
    if target_recall is not None:
        # Find the threshold that achieves at least target_recall with highest precision
        valid_indices = np.where(metrics['recall'] >= target_recall)[0]
        if len(valid_indices) == 0:
            raise ValueError(f"Cannot achieve recall of {target_recall}")
        optimal_idx = valid_indices[np.argmax(metrics['precision'][valid_indices])]
    elif target_precision is not None:
        # Find the threshold that achieves at least target_precision with highest recall
        valid_indices = np.where(metrics['precision'] >= target_precision)[0]
        if len(valid_indices) == 0:
            raise ValueError(f"Cannot achieve precision of {target_precision}")
        optimal_idx = valid_indices[np.argmax(metrics['recall'][valid_indices])]
    else:
        # Optimize for the specified metric
        if target_metric == 'f1':
            optimal_idx = np.argmax(metrics['f1'])
        elif target_metric == 'precision':
            optimal_idx = np.argmax(metrics['precision'])
        elif target_metric == 'recall':
            optimal_idx = np.argmax(metrics['recall'])
        elif target_metric == 'f2':
            f2_scores = (5 * metrics['precision'] * metrics['recall']) / \
                       (4 * metrics['precision'] + metrics['recall'] + 1e-9)
            optimal_idx = np.argmax(f2_scores)
        else:
            raise ValueError(f"Unsupported target_metric: {target_metric}")
    
    return metrics['thresholds'][optimal_idx], metrics[target_metric][optimal_idx]

def plot_precision_recall_vs_threshold(y_true: np.ndarray, y_scores: np.ndarray, output_dir: str):
    """Plot precision and recall as functions of the threshold."""
    import matplotlib.pyplot as plt
    import os
    
    metrics = evaluate_thresholds(y_true, y_scores)
    
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['thresholds'], metrics['precision'], "b--", label="Precision")
    plt.plot(metrics['thresholds'], metrics['recall'], "g-", label="Recall")
    plt.plot(metrics['thresholds'], metrics['f1'], "r-.", label="F1")
    
    # Add threshold that maximizes F1
    optimal_idx = np.argmax(metrics['f1'])
    plt.axvline(x=metrics['thresholds'][optimal_idx], color='k', linestyle='--', 
               label=f'Max F1 (t={metrics["thresholds"][optimal_idx]:.2f})')
    
    plt.xlabel("Threshold")
    plt.title("Precision, Recall, and F1 vs. Decision Threshold")
    plt.legend(loc="center left")
    plt.grid(True)
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "precision_recall_vs_threshold.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    
    return plot_path

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray, threshold: float, classification_weight: float) -> Dict[str, float]:
    """
    Calculate metrics for the ensemble model.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_scores: Predicted probability scores
        threshold: Decision threshold
        classification_weight: Weight for classification model
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, confusion_matrix
    )
    
    metrics = {'threshold': threshold}
    
    if y_true is not None and y_pred is not None and y_scores is not None:
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
        metrics['average_precision'] = average_precision_score(y_true, y_scores)
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
        metrics['classification_weight'] = classification_weight
        metrics['autoencoder_weight'] = 1.0 - classification_weight
    else:
        logger.warning("No validation data available. Skipping metrics calculation.")
        metrics = {'threshold': threshold, 'note': 'No validation data available'}
    
    return metrics

def train_ensemble(
    data_path: str,
    classification_model_path: str,
    autoencoder_model_path: str,
    output_dir: str,
    classification_weight: float = 0.7,
    threshold: Optional[float] = None,
    val_split: float = 0.2,
    test_split: float = 0.0,
    random_state: int = 42,
    tune_threshold: bool = True,  # Whether to tune the threshold
    target_recall: Optional[float] = None,  # Target recall to achieve
    target_precision: Optional[float] = None,  # Target precision to achieve
    apply_sampling: bool = False,  # Whether to apply advanced sampling
    sampling_technique: str = 'none',  # Sampling technique to use
    sampling_ratio: float = 0.5,  # Desired sampling ratio
    k_neighbors: int = 5  # Number of neighbors for sampling
) -> Tuple[FraudDetectionEnsemble, Dict[str, float]]:
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
        test_split: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        tune_threshold: Whether to tune the threshold
        target_recall: Target recall to achieve
        target_precision: Target precision to achieve
        apply_sampling: Whether to apply advanced sampling
        sampling_technique: Sampling technique to use
        sampling_ratio: Desired sampling ratio
        k_neighbors: Number of neighbors for sampling
        
    Returns:
        Tuple of (ensemble, metrics)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Preprocess data using unified data processor
    logger.info("Preprocessing data with unified data processor...")
    X_train, X_val, X_test, y_train, y_val, y_test, data_processor = preprocess_data(
        data_path, val_split, test_split, random_state,
        apply_sampling, sampling_technique, sampling_ratio, k_neighbors
    )
    
    # Log dataset shapes
    logger.info(f"Data shapes - X_train: {X_train.shape}, X_val: {X_val.shape if X_val is not None else 'N/A'}, X_test: {'N/A' if X_test is None or len(X_test) == 0 else X_test.shape}")
    
    # Load pre-trained models
    logger.info("Loading pre-trained models...")
    classification_model, autoencoder_model = load_models(classification_model_path, autoencoder_model_path)
    
    # Update autoencoder features in data processor if needed
    if hasattr(autoencoder_model, 'input_shape') and autoencoder_model.input_shape is not None:
        autoencoder_input_dim = autoencoder_model.input_shape[1]
        logger.info(f"Detected autoencoder input dimension: {autoencoder_input_dim}")
        
        # If we don't have autoencoder features defined yet, use a subset of classification features
        if data_processor.autoencoder_features is None:
            # Use all available features or limit to autoencoder input dimension
            if len(CLASSIFICATION_FEATURES) > autoencoder_input_dim:
                logger.warning(f"Autoencoder expects {autoencoder_input_dim} features but {len(CLASSIFICATION_FEATURES)} available")
                logger.warning("Using first N features that match autoencoder input dimension")
                data_processor.autoencoder_features = CLASSIFICATION_FEATURES[:autoencoder_input_dim]
            else:
                data_processor.autoencoder_features = CLASSIFICATION_FEATURES
    
    # Create ensemble model with data processor
    logger.info("Creating ensemble model with unified data processor...")
    ensemble = FraudDetectionEnsemble(
        classification_model=classification_model,
        autoencoder_model=autoencoder_model,
        classification_weight=classification_weight,
        threshold=threshold or 0.5,  # Default threshold, will be tuned later if requested
        data_processor=data_processor  # Pass the unified data processor
    )
    
    # Prepare autoencoder inputs using the data processor
    logger.info("Preparing autoencoder inputs...")
    try:
        # Check if we have autoencoder data from the data processor
        if hasattr(data_processor, 'autoencoder_data') and data_processor.autoencoder_data is not None:
            logger.info("Using autoencoder data from data processor")
            X_train_ae = data_processor.autoencoder_data['X_train']
        # If we have separate autoencoder features defined
        elif data_processor.autoencoder_features and data_processor.autoencoder_features != data_processor.classification_features:
            logger.info("Using separate feature set for autoencoder")
            # Load the data again but process it for autoencoder features
            df = data_processor.load_data(data_path)
            X_train_ae = data_processor.preprocess_features(df, data_processor.autoencoder_features)
            X_train_ae = data_processor.autoencoder_scaler.fit_transform(X_train_ae)
        else:
            # Use the same features as classification
            logger.info("Using same features for autoencoder as classification")
            X_train_ae = X_train
            
        # Fit reconstruction error scaler on training data
        logger.info("Fitting reconstruction error scaler...")
        # Pass the raw features to the ensemble's fit_reconstruction_scaler method
        # which will handle feature selection, reconstruction, and error calculation internally
        ensemble.fit_reconstruction_scaler(X_train_ae)
    except Exception as e:
        logger.error(f"Error preparing autoencoder inputs: {str(e)}")
        logger.warning("Falling back to using classification features for autoencoder")
        # Pass the raw features to the ensemble's fit_reconstruction_scaler method
        ensemble.fit_reconstruction_scaler(X_train)
    
    # Make predictions on validation data if available
    if X_val is not None and len(X_val) > 0:
        logger.info("Making predictions on validation data...")
        try:
            # Use the data processor for prediction
            y_pred_proba = ensemble.predict_proba(X_val)
            logger.info("Successfully used data processor for predictions")
        except Exception as e:
            logger.error(f"Error using data processor for predictions: {str(e)}")
            logger.warning("Falling back to direct prediction without data processor")
            # Direct prediction without data processor
            y_pred_proba = ensemble.predict_proba_raw(X_val)
    else:
        logger.warning("No validation data available. Using a default threshold of 0.5")
        threshold = 0.5
        y_pred_proba = None
    
    # Find optimal threshold if requested and validation data is available
    if (tune_threshold or threshold is None) and X_val is not None and y_val is not None and y_pred_proba is not None:
        logger.info("Finding optimal threshold...")
        # If target metrics are provided, find threshold that meets them
        if target_recall is not None or target_precision is not None:
            if target_recall is not None:
                logger.info(f"Finding threshold for target recall of {target_recall}")
                try:
                    threshold, recall = find_optimal_threshold(y_val, y_pred_proba, target_recall=target_recall)
                    logger.info(f"Found threshold {threshold:.4f} for target recall of {recall:.4f}")
                except ValueError as e:
                    logger.warning(f"Could not achieve target recall: {e}")
                    threshold, f1 = find_optimal_threshold(y_val, y_pred_proba, target_metric='f1')
                    logger.info(f"Using F1-optimal threshold instead: {threshold:.4f} (F1={f1:.4f})")
            else:
                logger.info(f"Finding threshold for target precision of {target_precision}")
                try:
                    threshold, precision = find_optimal_threshold(y_val, y_pred_proba, target_precision=target_precision)
                    logger.info(f"Found threshold {threshold:.4f} for target precision of {precision:.4f}")
                except ValueError as e:
                    logger.warning(f"Could not achieve target precision: {e}")
                    threshold, f1 = find_optimal_threshold(y_val, y_pred_proba, target_metric='f1')
                    logger.info(f"Using F1-optimal threshold instead: {threshold:.4f} (F1={f1:.4f})")
        else:
            # Otherwise maximize F1 score
            logger.info("Finding threshold that maximizes F1 score")
            threshold, f1 = find_optimal_threshold(y_val, y_pred_proba, target_metric='f1')
            logger.info(f"Threshold that maximizes F1: {threshold:.4f} (F1={f1:.4f})")
            
            # Log threshold for high recall (prioritize catching fraud)
            try:
                high_recall_threshold, recall = find_optimal_threshold(y_val, y_pred_proba, target_recall=0.8)
                logger.info(f"Threshold for 80% recall: {high_recall_threshold:.4f} (recall={recall:.4f})")
            except ValueError as e:
                logger.warning(f"Could not find threshold for 80% recall: {e}")
    elif threshold is None:
        # If we can't tune the threshold and none was provided, use default
        logger.warning("Cannot tune threshold without validation data. Using default threshold of 0.5")
        threshold = 0.5
    
    # Set the threshold
    ensemble.threshold = threshold
    logger.info(f"Using threshold: {threshold}")
    
    # Plot precision-recall curve if validation data is available
    if X_val is not None and y_val is not None and y_pred_proba is not None:
        try:
            plot_path = plot_precision_recall_vs_threshold(y_val, y_pred_proba, output_dir)
            logger.info(f"Saved precision-recall-threshold plot to {plot_path}")
        except Exception as e:
            logger.warning(f"Could not plot precision-recall curve: {str(e)}")
    
    # Make predictions and calculate metrics if validation data is available
    if X_val is not None and y_val is not None and y_pred_proba is not None:
        # Make predictions with the selected threshold
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate metrics
        metrics = calculate_metrics(y_val, y_pred, y_pred_proba, threshold, classification_weight)
        
        # Log metrics
        logger.info("\nEnsemble Model Metrics:")
        for metric, value in metrics.items():
            if metric != 'confusion_matrix':
                if isinstance(value, float):
                    logger.info(f"{metric}: {value:.4f}")
                else:
                    logger.info(f"{metric}: {value}")
    else:
        logger.warning("No validation data available. Skipping metrics calculation.")
        metrics = {'threshold': threshold, 'note': 'No validation data available'}
    
    # Save model with threshold, weights, and data processor
    logger.info("Saving ensemble model with data processor...")
    ensemble.threshold = threshold
    ensemble.classification_weight = classification_weight
    ensemble.autoencoder_weight = 1.0 - classification_weight
    
    # Save the data processor separately for easier inspection
    processor_dir = os.path.join(output_dir, 'data_processor')
    os.makedirs(processor_dir, exist_ok=True)
    data_processor.save(processor_dir)
    logger.info(f"Saved data processor to {processor_dir}")
    
    # Save the ensemble model
    ensemble.save(output_dir)
    logger.info(f"Saved ensemble model to {output_dir}")
    
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
                      help='Weight for classification model predictions (0-1)')
    parser.add_argument('--threshold', type=float, default=None,
                      help='Decision threshold for final prediction (0-1). If None, will find optimal threshold')
    parser.add_argument('--target-recall', type=float, default=None,
                      help='Target recall to achieve (overrides --threshold if specified)')
    parser.add_argument('--target-precision', type=float, default=None,
                      help='Target precision to achieve (overrides --threshold if specified)')
    parser.add_argument('--no-tune-threshold', action='store_false', dest='tune_threshold',
                      help='Disable threshold tuning and use the provided threshold')
    parser.add_argument('--val-split', type=float, default=0.2,
                      help='Fraction of data to use for validation')
    parser.add_argument('--random-state', type=int, default=42,
                      help='Random seed for reproducibility')
    parser.add_argument('--apply-sampling', action='store_true',
                      help='Apply advanced sampling techniques')
    parser.add_argument('--sampling-technique', type=str, default='none',
                      choices=['none', 'smote', 'adasyn', 'borderline_smote'],
                      help='Sampling technique to use')
    parser.add_argument('--sampling-ratio', type=float, default=0.5,
                      help='Desired ratio of minority to majority class after sampling')
    parser.add_argument('--k-neighbors', type=int, default=5,
                      help='Number of nearest neighbors for sampling techniques')
    parser.add_argument('--save-processor', action='store_true',
                      help='Save the data processor separately for inspection')
    
    args = parser.parse_args()
    
    # Train ensemble model
    ensemble, metrics = train_ensemble(
        data_path=args.data_path,
        classification_model_path=args.classification_model,
        autoencoder_model_path=args.autoencoder_model,
        output_dir=args.output_dir,
        classification_weight=args.classification_weight,
        threshold=args.threshold,
        tune_threshold=args.tune_threshold,
        val_split=args.val_split,
        random_state=args.random_state,
        target_recall=args.target_recall,
        target_precision=args.target_precision,
        apply_sampling=args.apply_sampling,
        sampling_technique=args.sampling_technique,
        sampling_ratio=args.sampling_ratio,
        k_neighbors=args.k_neighbors
    )  
    logger.info("Ensemble model training completed successfully!")

if __name__ == "__main__":
    main()
