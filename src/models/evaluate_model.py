"""
Script for evaluating trained fraud detection models and generating performance metrics and visualizations.
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import logging
import warnings
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, roc_auc_score, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)
import tensorflow as tf
import mlflow
import mlflow.tensorflow
import json

# Filter sklearn warnings to avoid excessive logging
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
# Specifically filter out the precision warning that's causing most of the noise
warnings.filterwarnings('ignore', message='Precision is ill-defined and being set to 0.0')
# Filter TensorFlow deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
# Filter TensorFlow model saving format warnings
warnings.filterwarnings('ignore', message='.*save_format.*is deprecated in Keras 3.*')
warnings.filterwarnings('ignore', message='.*HDF5 file.*is considered legacy.*')
# Filter optimizer variable mismatch warnings
warnings.filterwarnings('ignore', message='.*Skipping variable loading for optimizer.*')
# Filter absl warnings
warnings.filterwarnings('ignore', module='absl')
# Filter compiled metrics warnings
warnings.filterwarnings('ignore', message='.*Compiled the loaded model, but the compiled metrics.*')

# Configure logging for real-time output
os.makedirs('logs', exist_ok=True)

# Check if the root logger already has handlers to avoid duplicate logging
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/evaluate_model.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

# Get a logger specific to this module
logger = logging.getLogger('fraud_detection_evaluation')

# Force stdout to flush immediately
sys.stdout.reconfigure(line_buffering=True)

# Configure TensorFlow to use GPU if available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth for all GPUs to prevent TensorFlow from allocating all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("\n" + "="*70)
        print(f"🚀 GPU ACCELERATION ENABLED: Using {len(gpus)} GPU(s) for model evaluation")
        print("="*70 + "\n")
    except RuntimeError as e:
        print(f"Error configuring GPUs: {e}")
else:
    print("\n" + "="*70)
    print("⚠️ NO GPU DETECTED: Using CPU for model evaluation (this will be slower)")
    print("="*70 + "\n")

from fraud_model import compute_anomaly_scores

def load_model(model_path):
    """
    Load a trained TensorFlow model.
    
    Args:
        model_path (str): Path to the saved model
        
    Returns:
        Model: Loaded TensorFlow model
    """
    print(f"Loading model from {model_path}")
    
    # Check if the model path exists
    if not os.path.exists(model_path):
        # Try alternative formats if the exact path doesn't exist
        base_path = os.path.splitext(model_path)[0]
        alternative_paths = [
            f"{base_path}.keras",  # Native Keras format
            f"{base_path}.h5",     # HDF5 format
            f"{base_path}"         # Directory format
        ]
        
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                print(f"Using alternative model path: {alt_path}")
                model_path = alt_path
                break
    
    # Define custom objects for model loading
    custom_objects = {
        'mse': tf.keras.losses.MeanSquaredError(),
        'mean_squared_error': tf.keras.losses.MeanSquaredError()
    }
    
    # Try different loading approaches
    try:
        # First try with custom objects
        return tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    except Exception as e:
        print(f"Loading with custom objects failed: {str(e)}")
        try:
            # Then try standard loading
            return tf.keras.models.load_model(model_path)
        except Exception as e2:
            print(f"Standard loading failed: {str(e2)}")
            # As a last resort, try with compile=False
            print("Trying with compile=False...")
            try:
                return tf.keras.models.load_model(model_path, compile=False, custom_objects=custom_objects)
            except Exception as e3:
                print(f"All loading attempts failed: {str(e3)}")
                raise ValueError(f"Could not load model from {model_path}")

def load_test_data(file_path, model_path=None):
    """
    Load test data from a file.
    
    Args:
        file_path: Path to the test data file
        model_path: Path to the model directory (optional)
        
    Returns:
        X_test: Test features
        y_test: Test labels
    """
    print(f"Loading test data from {file_path}")

    # Use the same features as in training to ensure compatibility
    # First try to load feature names from the model directory if available
    if model_path:
        model_dir = os.path.dirname(model_path)
        feature_names_path = os.path.join(model_dir, 'feature_names.json')
        
        if os.path.exists(feature_names_path):
            try:
                with open(feature_names_path, 'r') as f:
                    core_features = json.load(f)
                print(f"Loaded {len(core_features)} features from feature_names.json")
            except Exception as e:
                print(f"Error loading feature names: {str(e)}")
                # Fallback to default features
                core_features = ['amount', 'spending_deviation_score', 'geo_anomaly_score', 'amount_log', 'velocity_score_norm', 'transaction_frequency']
        else:
            # These are the core numeric features used in training
            print("No feature_names.json found, using default features")
            core_features = ['amount', 'spending_deviation_score', 'geo_anomaly_score', 'amount_log', 'velocity_score_norm', 'transaction_frequency']
    else:
        # No model path provided, use default features
        print("No model path provided, using default features")
        core_features = ['amount', 'spending_deviation_score', 'geo_anomaly_score', 'amount_log', 'velocity_score_norm', 'transaction_frequency']
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Test data file not found: {file_path}")
    
    # Load data based on file extension
    if file_path.endswith('.parquet'):
        df = pd.read_parquet(file_path)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    print(f"Loaded data with {df.shape[0]} rows and {df.shape[1]} columns")
    
    # Use the same features as in training to ensure compatibility
    # First try to load feature names from the model directory if available
    model_dir = os.path.dirname(model_path)
    feature_names_path = os.path.join(model_dir, 'feature_names.json')
    
    if os.path.exists(feature_names_path):
        try:
            with open(feature_names_path, 'r') as f:
                core_features = json.load(f)
            print(f"Loaded {len(core_features)} features from feature_names.json")
        except Exception as e:
            print(f"Error loading feature names: {str(e)}")
            # Fallback to default features
            core_features = ['amount', 'spending_deviation_score', 'geo_anomaly_score', 'amount_log', 'velocity_score_norm', 'transaction_frequency']
    else:
        # These are the core numeric features used in training
        print("No feature_names.json found, using default features")
        core_features = ['amount', 'spending_deviation_score', 'geo_anomaly_score', 'amount_log', 'velocity_score_norm', 'transaction_frequency']
    
    # Check if all core features exist in the dataframe
    missing_features = [f for f in core_features if f not in df.columns]
    if missing_features:
        print(f"Warning: Missing features in test data: {missing_features}")
        print("Adding missing features with default values (zeros)")
        # Add missing features with default values
        for feature in missing_features:
            df[feature] = 0.0
    
    print(f"Using {len(core_features)} numeric features: {core_features}")
    
    # Prepare X and y
    if 'fraud_label' in df.columns:
        y_test = df['fraud_label'].values
        X_test = df[core_features].values
    elif 'is_fraud' in df.columns:
        # Convert boolean to int if needed
        y_test = df['is_fraud'].astype(int).values
        X_test = df[core_features].values
    else:
        # No target column, assume all data is X
        y_test = None
        X_test = df[core_features].values
    
    print(f"Prepared test data with {X_test.shape[0]} samples and {X_test.shape[1]} features")
    return X_test, y_test

def evaluate_classification_model(model, X_test, y_test, threshold=0.5, avg_transaction_amount=500, cost_fp_ratio=0.1, cost_fn_ratio=1.0):
    """
    Evaluate a classification model and generate performance metrics and visualizations.
    
    Args:
        model: Trained classification model
        X_test: Test features
        y_test: Test labels
        threshold: Classification threshold
        avg_transaction_amount: Average transaction amount for cost calculation
        cost_fp_ratio: Cost of false positive as ratio of transaction amount
        cost_fn_ratio: Cost of false negative as ratio of transaction amount
        
    Returns:
        dict: Dictionary of evaluation metrics including financial impact
    """
    print("Evaluating classification model...")
    
    # Get predictions
    try:
        y_pred_proba = model.predict(X_test).flatten()
        y_pred = (y_pred_proba >= threshold).astype(int)
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        print("Trying with a different approach...")
        # Try predicting one sample at a time
        y_pred_proba = np.zeros(len(X_test))
        for i in range(len(X_test)):
            try:
                y_pred_proba[i] = model.predict(X_test[i:i+1]).flatten()[0]
            except:
                y_pred_proba[i] = 0.0
        y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Check for single class in test data
    unique_classes = np.unique(y_test)
    print(f"Unique classes in test data: {unique_classes}")
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'threshold': threshold
    }
    
    # Only calculate these metrics if there's more than one class
    if len(unique_classes) > 1:
        try:
            metrics['precision'] = precision_score(y_test, y_pred)
            metrics['recall'] = recall_score(y_test, y_pred)
            metrics['f1_score'] = f1_score(y_test, y_pred)
            metrics['auc'] = roc_auc_score(y_test, y_pred_proba)
            metrics['average_precision'] = average_precision_score(y_test, y_pred_proba)
            
            # Calculate confusion matrix for cost-sensitive metrics
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            
            # Calculate financial costs
            fp_cost = fp * avg_transaction_amount * cost_fp_ratio  # Cost of investigation
            fn_cost = fn * avg_transaction_amount * cost_fn_ratio  # Cost of fraud loss
            total_cost = fp_cost + fn_cost
            
            # Add cost-sensitive metrics
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)
            metrics['true_positives'] = int(tp)
            metrics['true_negatives'] = int(tn)
            metrics['fp_cost'] = float(fp_cost)
            metrics['fn_cost'] = float(fn_cost)
            metrics['total_cost'] = float(total_cost)
            metrics['cost_per_transaction'] = float(total_cost / len(y_test))
            metrics['avg_transaction_amount'] = float(avg_transaction_amount)
            
            # Calculate cost savings compared to no model (all transactions legitimate)
            no_model_cost = sum(y_test) * avg_transaction_amount * cost_fn_ratio
            cost_savings = no_model_cost - total_cost
            metrics['cost_savings'] = float(cost_savings)
            metrics['cost_savings_percent'] = float((cost_savings / no_model_cost) * 100) if no_model_cost > 0 else 0.0
            
        except Exception as e:
            print(f"Warning: Could not calculate some metrics: {str(e)}")
    else:
        print("Warning: Only one class present in test data. AUC cannot be calculated.")
        # Add class-specific metrics for the single class
        class_label = unique_classes[0]
        metrics[f'{class_label}_precision'] = 1.0 if np.all(y_pred == class_label) else 0.0
        metrics[f'{class_label}_recall'] = 1.0
        metrics[f'{class_label}_f1-score'] = 1.0 if np.all(y_pred == class_label) else 0.0
        metrics[f'{class_label}_support'] = float(len(y_test))
    
    # Get classification report as a dict
    try:
        report = classification_report(y_test, y_pred, output_dict=True)
        # Add report metrics to our metrics dict
        for class_name, class_metrics in report.items():
            if isinstance(class_metrics, dict):
                for metric_name, value in class_metrics.items():
                    metrics[f'{class_name}_{metric_name}'] = value
    except Exception as e:
        print(f"Warning: Could not generate classification report: {str(e)}")
    
    # Print metrics
    print("\nClassification Performance Metrics:")
    for metric, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"{metric}: {value}")
    
    # Generate visualizations only if there's more than one class
    if len(unique_classes) > 1:
        try:
            generate_classification_visualizations(y_test, y_pred, y_pred_proba)
            
            # Generate cost-sensitive visualizations
            generate_cost_sensitive_visualizations(y_test, y_pred_proba, avg_transaction_amount, 
                                                  cost_fp_ratio, cost_fn_ratio)
        except Exception as e:
            print(f"Warning: Could not generate visualizations: {str(e)}")
    
    return metrics

def evaluate_autoencoder_model(model, X_test, y_test, threshold=None, percentile=95):
    """
    Evaluate an autoencoder model for anomaly detection.
    
    Args:
        model: Trained autoencoder model
        X_test: Test features
        y_test: Test labels
        threshold: Anomaly score threshold (if None, determined from data)
        percentile: Percentile for threshold determination
        
    Returns:
        dict: Dictionary of evaluation metrics and anomaly scores
    """
    print("Evaluating autoencoder model...")
    
    # Compute anomaly scores (reconstruction errors)
    anomaly_scores = compute_anomaly_scores(model, X_test)
    
    # Determine threshold if not provided
    if threshold is None:
        # Use specified percentile of non-fraud scores as threshold
        non_fraud_indices = (y_test == 0)
        non_fraud_scores = anomaly_scores[non_fraud_indices]
        threshold = np.percentile(non_fraud_scores, percentile)
        print(f"Using {percentile}th percentile as threshold: {threshold:.4f}")
    
    # Classify based on threshold
    y_pred = (anomaly_scores >= threshold).astype(int)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'threshold': threshold,
        'percentile': percentile
    }
    
    # Print metrics
    print("\nAnomaly Detection Performance Metrics:")
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Generate visualizations
    generate_autoencoder_visualizations(y_test, y_pred, anomaly_scores, threshold)
    
    return metrics, anomaly_scores

def generate_classification_visualizations(y_true, y_pred, y_pred_proba, output_dir='../../results/figures'):
    """
    Generate visualizations for classification model evaluation.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
        output_dir: Directory to save figures
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Precision-Recall Curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    ap = average_precision_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'AP = {ap:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")

def generate_cost_sensitive_visualizations(y_true, y_pred_proba, avg_transaction_amount=500, 
                                cost_fp_ratio=0.1, cost_fn_ratio=1.0, output_dir='../../results/figures'):
    """
    Generate cost-sensitive visualizations for fraud detection model evaluation.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        avg_transaction_amount: Average transaction amount
        cost_fp_ratio: Cost of false positive as ratio of transaction amount
        cost_fn_ratio: Cost of false negative as ratio of transaction amount
        output_dir: Directory to save figures
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate costs at different thresholds
    thresholds = np.linspace(0.01, 0.99, 99)
    costs = []
    fp_costs = []
    fn_costs = []
    f1_scores = []
    recalls = []
    precisions = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculate costs
        fp_cost = fp * avg_transaction_amount * cost_fp_ratio
        fn_cost = fn * avg_transaction_amount * cost_fn_ratio
        total_cost = fp_cost + fn_cost
        
        costs.append(total_cost)
        fp_costs.append(fp_cost)
        fn_costs.append(fn_cost)
        
        # Calculate performance metrics
        f1 = f1_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        
        f1_scores.append(f1)
        recalls.append(recall)
        precisions.append(precision)
    
    # Plot costs vs threshold
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, costs, label='Total Cost')
    plt.plot(thresholds, fp_costs, label='False Positive Cost')
    plt.plot(thresholds, fn_costs, label='False Negative Cost')
    plt.xlabel('Threshold')
    plt.ylabel('Cost ($)')
    plt.title('Financial Impact vs. Classification Threshold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'cost_vs_threshold.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Find optimal threshold based on cost
    optimal_threshold_idx = np.argmin(costs)
    optimal_threshold = thresholds[optimal_threshold_idx]
    min_cost = costs[optimal_threshold_idx]
    
    # Plot metrics vs threshold with cost-optimal point highlighted
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores, label='F1 Score')
    plt.plot(thresholds, recalls, label='Recall')
    plt.plot(thresholds, precisions, label='Precision')
    plt.axvline(x=optimal_threshold, color='r', linestyle='--', label=f'Cost-optimal threshold: {optimal_threshold:.2f}')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Performance Metrics vs. Threshold with Cost-Optimal Point')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'metrics_with_cost_optimal.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate cost savings at different recall levels
    recall_levels = np.linspace(0.5, 1.0, 11)
    savings = []
    thresholds_for_recall = []
    
    for target_recall in recall_levels:
        # Find threshold that achieves target recall
        recall_diffs = [abs(r - target_recall) for r in recalls]
        closest_idx = np.argmin(recall_diffs)
        threshold_for_recall = thresholds[closest_idx]
        thresholds_for_recall.append(threshold_for_recall)
        
        # Calculate cost savings at this threshold
        y_pred = (y_pred_proba >= threshold_for_recall).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculate costs
        total_cost = (fp * avg_transaction_amount * cost_fp_ratio) + (fn * avg_transaction_amount * cost_fn_ratio)
        no_model_cost = sum(y_true) * avg_transaction_amount * cost_fn_ratio
        saving = no_model_cost - total_cost
        savings.append(saving)
    
    # Plot cost savings vs recall
    plt.figure(figsize=(10, 6))
    plt.plot(recall_levels, savings)
    plt.xlabel('Recall (Fraud Detection Rate)')
    plt.ylabel('Cost Savings ($)')
    plt.title('Cost Savings vs. Fraud Detection Rate')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'cost_savings_vs_recall.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Cost-sensitive visualizations saved to {output_dir}")
    print(f"Cost-optimal threshold: {optimal_threshold:.4f} with minimum cost: ${min_cost:.2f}")

def generate_autoencoder_visualizations(y_true, y_pred, anomaly_scores, threshold, output_dir='../../results/figures'):
    """
    Generate visualizations for autoencoder model evaluation.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        anomaly_scores: Anomaly scores (reconstruction errors)
        threshold: Threshold used for classification
        output_dir: Directory to save figures
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(output_dir, 'autoencoder_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Anomaly Score Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(anomaly_scores[y_true == 0], label='Normal', alpha=0.5, kde=True)
    sns.histplot(anomaly_scores[y_true == 1], label='Fraud', alpha=0.5, kde=True)
    plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.4f}')
    plt.xlabel('Anomaly Score (Reconstruction Error)')
    plt.ylabel('Count')
    plt.title('Anomaly Score Distribution')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'anomaly_score_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # ROC Curve for Anomaly Scores
    fpr, tpr, thresholds = roc_curve(y_true, anomaly_scores)
    auc = roc_auc_score(y_true, anomaly_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Anomaly Scores)')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(output_dir, 'autoencoder_roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")

def save_metrics(metrics, model_type, output_dir='../../results/metrics'):
    """
    Save evaluation metrics to a JSON file.
    
    Args:
        metrics: Dictionary of evaluation metrics
        model_type: Type of model ('classification' or 'autoencoder')
        output_dir: Directory to save metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert numpy types to Python native types for JSON serialization
    metrics_json = {}
    for k, v in metrics.items():
        if isinstance(v, np.ndarray):
            metrics_json[k] = v.tolist()
        elif isinstance(v, np.generic):
            metrics_json[k] = v.item()
        else:
            metrics_json[k] = v
    
    # Save to JSON file
    output_file = os.path.join(output_dir, f'{model_type}_metrics.json')
    with open(output_file, 'w') as f:
        json.dump(metrics_json, f, indent=4)
    
    print(f"Metrics saved to {output_file}")

def main():
    """
    Main function to run the model evaluation.
    """
    parser = argparse.ArgumentParser(description='Evaluate a trained fraud detection model')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the saved model')
    parser.add_argument('--test-data', type=str, required=True,
                        help='Path to the test data file')
    parser.add_argument('--model-type', type=str, choices=['classification', 'autoencoder'],
                        required=True, help='Type of model to evaluate')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Classification threshold (for classification) or anomaly threshold (for autoencoder)')
    parser.add_argument('--percentile', type=float, default=95,
                        help='Percentile for threshold determination (autoencoder only)')
    parser.add_argument('--output-dir', type=str, default='../../results',
                        help='Directory to save evaluation results')
    
    # Add cost-sensitive evaluation parameters
    parser.add_argument('--avg-transaction-amount', type=float, default=500,
                        help='Average transaction amount for cost calculation')
    parser.add_argument('--cost-fp-ratio', type=float, default=0.1,
                        help='Cost of false positive as ratio of transaction amount (investigation cost)')
    parser.add_argument('--cost-fn-ratio', type=float, default=1.0,
                        help='Cost of false negative as ratio of transaction amount (fraud loss)')
    
    # Add GPU-related arguments to match the pipeline's parameters
    parser.add_argument('--disable-gpu', action='store_true',
                        help='Disable GPU usage even if available')
    parser.add_argument('--single-gpu', action='store_true',
                        help='Use only a single GPU even if multiple are available')
    parser.add_argument('--memory-growth', action='store_true',
                        help='Enable memory growth for GPUs to prevent TensorFlow from allocating all memory')
    args = parser.parse_args()
    
    # Set up MLflow
    mlflow.set_experiment("Fraud_Detection_Evaluation")
    
    # Load model and test data
    model = load_model(args.model_path)
    X_test, y_test = load_test_data(args.test_data, args.model_path)
    
    # Run evaluation based on model type
    with mlflow.start_run(run_name=f"evaluate_{args.model_type}"):
        if args.model_type == 'classification':
            threshold = 0.5 if args.threshold is None else args.threshold
            metrics = evaluate_classification_model(
                model, X_test, y_test, threshold,
                avg_transaction_amount=args.avg_transaction_amount,
                cost_fp_ratio=args.cost_fp_ratio,
                cost_fn_ratio=args.cost_fn_ratio
            )
            
            # Log metrics to MLflow
            for metric, value in metrics.items():
                mlflow.log_metric(metric, value)
            
            # Save metrics to file
            save_metrics(metrics, args.model_type, os.path.join(args.output_dir, 'metrics'))
            
            # Log cost-related parameters
            mlflow.log_param('avg_transaction_amount', args.avg_transaction_amount)
            mlflow.log_param('cost_fp_ratio', args.cost_fp_ratio)
            mlflow.log_param('cost_fn_ratio', args.cost_fn_ratio)
            
        elif args.model_type == 'autoencoder':
            try:
                metrics_result = evaluate_autoencoder_model(
                    model, X_test, y_test, args.threshold, args.percentile
                )
                
                # Handle different return formats
                if isinstance(metrics_result, tuple) and len(metrics_result) == 2:
                    metrics, anomaly_scores = metrics_result
                else:
                    metrics = metrics_result
                    anomaly_scores = None
                
                # Log metrics to MLflow
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(metric, value)
                
                # Save metrics to file
                save_metrics(metrics, args.model_type, os.path.join(args.output_dir, 'metrics'))
            except Exception as e:
                print(f"Error during autoencoder evaluation: {str(e)}")
                print("Continuing with pipeline...")
                metrics = {'error': str(e)}
                save_metrics(metrics, args.model_type, os.path.join(args.output_dir, 'metrics'))
        
        # Log figures to MLflow if they exist
        figures_dir = os.path.join(args.output_dir, 'figures')
        if os.path.exists(figures_dir):
            try:
                for figure_file in os.listdir(figures_dir):
                    if figure_file.endswith('.png'):
                        mlflow.log_artifact(os.path.join(figures_dir, figure_file), 'figures')
            except Exception as e:
                print(f"Warning: Could not log figures to MLflow: {str(e)}")
        else:
            print(f"Figures directory {figures_dir} does not exist. Skipping figure logging.")
            # Create the directory for future use
            os.makedirs(figures_dir, exist_ok=True)

if __name__ == "__main__":
    main()
