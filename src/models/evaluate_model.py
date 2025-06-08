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
    confusion_matrix, classification_report, auc
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

# Try different import approaches to handle both module and script execution
try:
    # When imported as a module
    from src.models.fraud_model import compute_anomaly_scores
except ModuleNotFoundError:
    try:
        # When run as a script from project root
        import sys
        import os
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
        from src.models.fraud_model import compute_anomaly_scores
    except ModuleNotFoundError:
        # When run as a script from models directory
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
    Load and prepare test data for model evaluation.
    
    Args:
        file_path: Path to the test data CSV file
        model_path: Path to the model directory (to find feature names)
        
    Returns:
        tuple: (X_test, y_test, feature_names) prepared for model evaluation
    """
    print(f"Loading test data from {file_path}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Test data file not found: {file_path}")
    
    # Load test data based on file extension
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == '.parquet' or file_path.endswith('.parquet'):
        try:
            df = pd.read_parquet(file_path)
        except Exception as e:
            print(f"Error reading parquet file: {str(e)}")
            # If the parquet file is actually a directory (Spark format), try reading from it
            if os.path.isdir(file_path):
                parquet_files = [os.path.join(file_path, f) for f in os.listdir(file_path) 
                                if f.endswith('.parquet') and not f.startswith('.')]
                if parquet_files:
                    df = pd.read_parquet(parquet_files[0])
                else:
                    raise FileNotFoundError(f"No parquet files found in directory: {file_path}")
            else:
                raise
    elif file_extension == '.csv':
        df = pd.read_csv(file_path)
    elif file_extension == '.json':
        df = pd.read_json(file_path)
    else:
        # Try to infer the file format
        try:
            df = pd.read_csv(file_path)
        except Exception:
            try:
                df = pd.read_parquet(file_path)
            except Exception:
                try:
                    df = pd.read_json(file_path)
                except Exception as e:
                    raise ValueError(f"Unsupported file format or corrupted file: {file_path}. Error: {str(e)}")
    
    # Default features (fallback if feature names file not found)
    default_features = [
        'amount', 'oldbalanceOrg', 'newbalanceOrig', 
        'oldbalanceDest', 'newbalanceDest', 'hour'
    ]
    
    # Try to load feature names from the model directory
    core_features = default_features.copy()
    if model_path:
        model_dir = os.path.dirname(model_path)
        model_filename = os.path.basename(model_path)
        
        # Try multiple approaches to find the feature names file
        feature_names_paths = [
            os.path.join(model_dir, 'feature_names.json'),
            os.path.join(model_dir, model_filename.replace('_model.keras', '_feature_names.json')),
            os.path.join(model_dir, model_filename.replace('model.keras', 'feature_names.json'))
        ]
        
        # Add any timestamp-based feature names files
        try:
            timestamp_files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) 
                              if f.endswith('_feature_names.json')]
            feature_names_paths.extend(timestamp_files)
        except Exception as e:
            print(f"Warning: Error accessing model directory: {str(e)}")
        
        # Try each potential path
        for feature_path in feature_names_paths:
            if os.path.exists(feature_path):
                try:
                    with open(feature_path, 'r') as f:
                        core_features = json.load(f)
                    print(f"Loaded feature names from {feature_path}")
                    break
                except Exception as e:
                    print(f"Error loading feature names from {feature_path}: {str(e)}")
        
        if core_features == default_features:
            print(f"Warning: Could not find feature names file. Using default features: {default_features}")
    else:
        print("No model path provided, using default features")
    
    # Ensure all required features exist in the dataframe
    for feature in core_features:
        if feature not in df.columns:
            print(f"Warning: Feature '{feature}' not found in test data. Adding with appropriate default values.")
            
            # Add missing feature with appropriate default values
            if feature == 'hour':
                # Extract hour from timestamp if available
                if 'timestamp' in df.columns:
                    try:
                        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
                    except:
                        df['hour'] = 12  # Default to noon
                else:
                    df['hour'] = 12  # Default to noon
            elif 'balance' in feature.lower():
                # Use median of other balance columns if available
                balance_cols = [col for col in df.columns if 'balance' in col.lower()]
                if balance_cols:
                    df[feature] = df[balance_cols].median(axis=1)
                else:
                    df[feature] = 0
            elif feature == 'amount':
                if 'amount' in df.columns:
                    # If there's a column with the same name but different case
                    amount_cols = [col for col in df.columns if col.lower() == 'amount']
                    if amount_cols:
                        df[feature] = df[amount_cols[0]]
                    else:
                        df[feature] = 0  # Default amount
                else:
                    df[feature] = 0  # Default amount
            elif 'delta' in feature.lower() or 'ratio' in feature.lower():
                # For derived features, try to compute them if possible
                if 'amount_delta' == feature and 'amount' in df.columns and 'oldbalanceOrg' in df.columns:
                    df[feature] = df['amount'] - df['oldbalanceOrg']
                elif 'amount_ratio' == feature and 'amount' in df.columns and 'oldbalanceOrg' in df.columns:
                    # Avoid division by zero
                    df[feature] = df['amount'] / df['oldbalanceOrg'].replace(0, 1)
                else:
                    df[feature] = 0
            elif 'time_since' in feature.lower():
                # Time-based features default to 0 (no history)
                df[feature] = 0
            elif 'avg_' in feature.lower() or 'max_' in feature.lower():
                # Aggregate features default to the current value if available
                base_feature = feature.split('_')[1] if '_' in feature else 'amount'
                if base_feature in df.columns:
                    df[feature] = df[base_feature]
                else:
                    df[feature] = 0
            else:
                df[feature] = 0  # Default for other features
    
    # Prepare X and y
    if 'fraud_label' in df.columns:
        y_test = df['fraud_label'].values
        X_test = df[core_features].copy()
    elif 'is_fraud' in df.columns:
        # Convert boolean to int if needed
        y_test = df['is_fraud'].astype(int).values
        X_test = df[core_features].copy()
    else:
        # No target column, assume all data is X
        y_test = None
        X_test = df[core_features].copy()
    
    # Handle NaN values in the features
    if X_test.isna().any().any():
        print(f"Warning: NaN values detected in {X_test.isna().sum().sum()} cells across {X_test.isna().any().sum()} features")
        print("Handling NaN values in features...")
        
        # For each feature with NaN values, apply appropriate handling
        for feature in X_test.columns[X_test.isna().any()]:
            nan_count = X_test[feature].isna().sum()
            nan_percent = (nan_count / len(X_test)) * 100
            print(f"  - '{feature}': {nan_count} NaN values ({nan_percent:.2f}%)")
            
            # Strategy 1: Fill with median for numeric features
            median_value = X_test[feature].median()
            if pd.notna(median_value):
                X_test[feature].fillna(median_value, inplace=True)
                print(f"    Filled NaN values with median: {median_value}")
            elif 'time_since' in feature.lower() or 'delta' in feature.lower() or 'ratio' in feature.lower():
                # For time-based or derived features, use 0 as a reasonable default
                X_test[feature].fillna(0, inplace=True)
                print(f"    Filled NaN values with 0 (appropriate for this feature type)")
            elif 'avg_' in feature.lower() or 'max_' in feature.lower():
                # For aggregate features, try to use the base feature value
                base_feature = feature.split('_')[1] if '_' in feature else None
                if base_feature in X_test.columns:
                    X_test[feature] = X_test[feature].fillna(X_test[base_feature])
                    print(f"    Filled NaN values with corresponding base feature values")
                else:
                    X_test[feature].fillna(0, inplace=True)
                    print(f"    Filled NaN values with 0 (fallback)")
            else:
                # Strategy 2: If median is also NaN, use 0
                X_test[feature].fillna(0, inplace=True)
                print(f"    Filled NaN values with 0 (median was also NaN)")
    
    # Verify no NaN values remain
    if X_test.isna().any().any():
        # If NaNs still exist after targeted handling, use a final fallback
        print("Warning: NaN values still present after initial handling, applying global fillna")
        X_test.fillna(0, inplace=True)
    
    print(f"Prepared test data with {X_test.shape[0]} samples and {X_test.shape[1]} features")
    
    # Convert to numpy arrays for model compatibility
    X_test_array = X_test.values
    
    return X_test_array, y_test, core_features

def evaluate_classification_model(model, X_test, y_test, feature_names=None, threshold=None, avg_transaction_amount=500, cost_fp_ratio=0.1, cost_fn_ratio=1.0, output_dir='results/figures'):
    """
    Evaluate a classification model for fraud detection.
    
    Args:
        model: Trained classification model
        X_test: Test features
        y_test: Test labels
        feature_names: List of feature names for importance analysis
        threshold: Classification threshold (default: 0.5)
        avg_transaction_amount: Average transaction amount for cost calculation
        cost_fp_ratio: Cost ratio for false positives
        cost_fn_ratio: Cost ratio for false negatives
        output_dir: Directory to save visualizations
        
    Returns:
        dict: Evaluation metrics
    """
    print("Evaluating classification model...")
    
    # Check if y_test is None or empty
    if y_test is None or len(y_test) == 0:
        print("Warning: No labels in test data, skipping evaluation")
        return {"error": "No labels in test data"}
    
    # Get unique classes in test set
    unique_classes = np.unique(y_test)
    print(f"Unique classes in test set: {unique_classes}")
    
    # Handle case where there's only one class in test set
    if len(unique_classes) == 1:
        print(f"Warning: Only one class ({unique_classes[0]}) in test set")
        if unique_classes[0] == 0:
            print("Test set contains only non-fraud cases")
        else:
            print("Test set contains only fraud cases")
    
    # Get predictions
    try:
        # Try batch prediction first
        y_pred_proba = model.predict(X_test)
        
        # Check if output is multi-dimensional (for multi-class)
        if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
            # Get probability of positive class (fraud)
            y_pred_proba = y_pred_proba[:, 1]
    except Exception as e:
        print(f"Batch prediction failed: {str(e)}. Trying sample-by-sample prediction.")
        # Fall back to sample-by-sample prediction
        y_pred_proba = np.array([model.predict(x.reshape(1, -1))[0] for x in X_test])
    
    # Find optimal threshold if not provided
    if threshold is None and len(unique_classes) > 1:
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        # Find threshold that maximizes F1 score
        f1_scores = []
        for p, r in zip(precision, recall):
            if p + r == 0:
                f1_scores.append(0)
            else:
                f1_scores.append(2 * p * r / (p + r))
        
        if len(thresholds) > 0:
            # Add the last threshold (precision-recall curve doesn't include it)
            thresholds = np.append(thresholds, 1.0)
            best_idx = np.argmax(f1_scores)
            threshold = thresholds[best_idx]
            print(f"Optimal threshold: {threshold:.4f} (F1: {f1_scores[best_idx]:.4f})")
        else:
            # Fallback if no thresholds found
            threshold = 0.5
            print(f"No optimal threshold found, using default: {threshold}")
    else:
        # Use default threshold if not provided or only one class in test set
        threshold = 0.5 if threshold is None else threshold
        print(f"Using threshold: {threshold}")
    
    # Apply threshold to get binary predictions
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate metrics
    metrics = {}
    
    # Basic classification metrics
    metrics['accuracy'] = accuracy_score(y_test, y_pred)
    
    # Handle case where predictions are all one class
    if len(np.unique(y_pred)) == 1:
        print(f"Warning: All predictions are class {np.unique(y_pred)[0]}")
        if np.unique(y_pred)[0] == 0:
            # All predicted as non-fraud
            metrics['precision'] = 0.0
            metrics['recall'] = 0.0
            metrics['f1_score'] = 0.0
        else:
            # All predicted as fraud
            metrics['precision'] = float(np.mean(y_test == 1))
            metrics['recall'] = 1.0
            metrics['f1_score'] = 2 * metrics['precision'] / (metrics['precision'] + 1) if metrics['precision'] > 0 else 0.0
    else:
        # Normal case with mixed predictions
        metrics['precision'] = precision_score(y_test, y_pred)
        metrics['recall'] = recall_score(y_test, y_pred)
        metrics['f1_score'] = f1_score(y_test, y_pred)
    
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
            generate_classification_visualizations(y_true=y_test, y_pred=y_pred, y_pred_proba=y_pred_proba, output_dir=output_dir)
            
            # Generate cost-sensitive visualizations
            generate_cost_sensitive_visualizations(y_true=y_test, y_pred_proba=y_pred_proba, avg_transaction_amount=avg_transaction_amount, 
                                              cost_fp_ratio=cost_fp_ratio, cost_fn_ratio=cost_fn_ratio, output_dir=output_dir)
        except Exception as e:
            print(f"Warning: Could not generate visualizations: {str(e)}")
    
    return metrics

def evaluate_autoencoder_model(model, X_test, y_test, feature_names=None, threshold=None, percentile=90, optimize_threshold=True, output_dir='results/figures'):
    """
    Evaluate an autoencoder model for anomaly detection.
    
    Args:
        model: Trained autoencoder model
        X_test: Test features
        y_test: Test labels
        feature_names: List of feature names for importance analysis
        threshold: Anomaly score threshold (if None, determined from data)
        percentile: Percentile for threshold determination
        optimize_threshold: Whether to optimize the threshold for best F1 score
        output_dir: Directory to save visualizations
        
    Returns:
        dict: Dictionary of evaluation metrics and anomaly scores
    """
    print("Evaluating autoencoder model...")
    
    # Compute anomaly scores (reconstruction errors)
    anomaly_scores = compute_anomaly_scores(model, X_test)
    
    # Create output directory for visualizations
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine threshold if not provided
    if threshold is None:
        # Separate scores for normal (non-fraud) transactions
        non_fraud_indices = np.where(y_test == 0)[0]
        fraud_indices = np.where(y_test == 1)[0]
        
        if len(non_fraud_indices) == 0:
            print("Warning: No non-fraud samples in test data, using median of all scores")
            threshold = np.median(anomaly_scores)
        else:
            non_fraud_scores = anomaly_scores[non_fraud_indices]
            fraud_scores = anomaly_scores[fraud_indices] if len(fraud_indices) > 0 else np.array([])
            
            # Plot distribution of reconstruction errors
            plt.figure(figsize=(10, 6))
            plt.hist(non_fraud_scores, bins=50, alpha=0.5, label='Non-Fraud')
            if len(fraud_scores) > 0:
                plt.hist(fraud_scores, bins=50, alpha=0.5, label='Fraud')
            plt.xlabel('Reconstruction Error')
            plt.ylabel('Count')
            plt.title('Distribution of Reconstruction Errors')
            plt.legend()
            plt.savefig(os.path.join(output_dir, 'reconstruction_error_dist.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            if optimize_threshold:
                print("Optimizing anomaly threshold...")
                best_f1 = 0
                best_threshold = None
                best_percentile = None
                all_results = []
                
                # Try different percentiles to find optimal threshold
                for p in range(75, 100):
                    temp_threshold = np.percentile(non_fraud_scores, p)
                    temp_preds = (anomaly_scores >= temp_threshold).astype(int)
                    temp_precision = precision_score(y_test, temp_preds)
                    temp_recall = recall_score(y_test, temp_preds)
                    temp_f1 = f1_score(y_test, temp_preds)
                    
                    all_results.append((p, temp_threshold, temp_precision, temp_recall, temp_f1))
                    
                    if temp_f1 > best_f1:
                        best_f1 = temp_f1
                        best_threshold = temp_threshold
                        best_percentile = p
                
                # Plot threshold optimization results
                plt.figure(figsize=(12, 8))
                percentiles, thresholds, precisions, recalls, f1s = zip(*all_results)
                plt.plot(percentiles, precisions, label='Precision')
                plt.plot(percentiles, recalls, label='Recall')
                plt.plot(percentiles, f1s, label='F1 Score')
                plt.axvline(x=best_percentile, color='r', linestyle='--', label=f'Best Percentile: {best_percentile}')
                plt.xlabel('Percentile')
                plt.ylabel('Score')
                plt.title('Threshold Optimization')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(output_dir, 'threshold_optimization.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
                if best_threshold is not None:
                    threshold = best_threshold
                    print(f"Optimal threshold: {threshold:.4f} (percentile: {best_percentile}, F1: {best_f1:.4f})")
                else:
                    # Fallback to default percentile
                    threshold = np.percentile(non_fraud_scores, percentile)
                    print(f"Using default percentile threshold: {threshold:.4f} (percentile: {percentile})")
            else:
                # Use specified percentile
                threshold = np.percentile(non_fraud_scores, percentile)
                print(f"Using specified percentile threshold: {threshold:.4f} (percentile: {percentile})")
    else:
        print(f"Using provided threshold: {threshold}")
    
    # Apply threshold to get predictions
    y_pred = (anomaly_scores >= threshold).astype(int)
    
    # Calculate metrics
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_test, y_pred)
    
    # Handle case where predictions are all one class
    if len(np.unique(y_pred)) == 1:
        print(f"Warning: All predictions are class {np.unique(y_pred)[0]}")
        if np.unique(y_pred)[0] == 0:
            # All predicted as non-fraud
            metrics['precision'] = 0.0
            metrics['recall'] = 0.0
            metrics['f1_score'] = 0.0
        else:
            # All predicted as fraud
            metrics['precision'] = float(np.mean(y_test == 1))
            metrics['recall'] = 1.0
            metrics['f1_score'] = 2 * metrics['precision'] / (metrics['precision'] + 1) if metrics['precision'] > 0 else 0.0
    else:
        # Normal case with mixed predictions
        metrics['precision'] = precision_score(y_test, y_pred)
        metrics['recall'] = recall_score(y_test, y_pred)
        metrics['f1_score'] = f1_score(y_test, y_pred)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    metrics['true_positives'] = int(tp)
    
    # Add threshold information
    metrics['threshold'] = float(threshold)
    metrics['threshold_percentile'] = float(percentile) if percentile else 0.0
    
    # Generate confusion matrix visualization
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (Autoencoder)')
    plt.savefig(os.path.join(output_dir, 'autoencoder_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Analyze feature importance if feature names are provided
    if feature_names is not None:
        try:
            # For autoencoders, we can analyze which features contribute most to reconstruction error
            feature_errors = []
            
            # Get reconstructed data
            X_reconstructed = model.predict(X_test)
            
            # Calculate error for each feature
            for i, feature in enumerate(feature_names):
                feature_error = np.mean(np.abs(X_test[:, i] - X_reconstructed[:, i]))
                feature_errors.append((feature, feature_error))
            
            # Sort by error
            feature_errors.sort(key=lambda x: x[1], reverse=True)
            
            # Add to metrics
            metrics['feature_importance'] = {k: float(v) for k, v in feature_errors[:10]}
            
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            features, errors = zip(*feature_errors[:20])  # Top 20 features
            plt.barh(range(len(features)), errors, align='center')
            plt.yticks(range(len(features)), features)
            plt.xlabel('Mean Reconstruction Error')
            plt.title('Feature Importance by Reconstruction Error')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'autoencoder_feature_importance.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Also try permutation importance if we have labels
            if len(np.unique(y_test)) > 1:
                importance_scores = analyze_feature_importance(model, X_test, y_test, feature_names, 
                                                              os.path.join(output_dir, 'autoencoder_permutation'))
                if importance_scores:
                    # Add top 10 most important features to metrics
                    sorted_features = sorted(importance_scores.items(), key=lambda x: abs(x[1]), reverse=True)
                    metrics['permutation_importance'] = {k: float(v) for k, v in sorted_features[:10]}
        except Exception as e:
            print(f"Warning: Feature importance analysis failed: {str(e)}")
    
    # Print summary
    print(f"Autoencoder Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1 Score: {metrics['f1_score']:.4f}")
    print(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"  Threshold: {threshold:.4f}")
    
    return metrics, anomaly_scores

def analyze_feature_importance(model, X_test, y_test, feature_names, output_dir='results/figures'):
    """
    Analyze feature importance for fraud detection models.
    
    Args:
        model: Trained model (classification or autoencoder)
        X_test: Test features
        y_test: Test labels
        feature_names: List of feature names
        output_dir: Directory to save visualizations
    
    Returns:
        dict: Feature importance scores
    """
    print("Analyzing feature importance...")
    os.makedirs(output_dir, exist_ok=True)
    
    # For classification models with built-in feature importance
    if hasattr(model, 'feature_importances_'):
        # Random Forest, Gradient Boosting, etc.
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Create feature importance dict
        importance_dict = {feature_names[i]: float(importances[i]) for i in range(len(feature_names))}
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importance (Built-in)')
        plt.bar(range(len(importances)), importances[indices], align='center')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance_builtin.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # For neural network models (classification and autoencoder)
    # Use permutation importance
    try:
        # For classification models
        if hasattr(model, 'predict_proba'):
            def score_func(X):
                return model.predict_proba(X)[:, 1]
        # For autoencoder models
        else:
            def score_func(X):
                preds = model.predict(X)
                return np.mean(np.square(X - preds), axis=1)
        
        # Calculate permutation importance
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
        baseline_score = score_func(X_test)
        importance_scores = {}
        
        for feature in feature_names:
            # Create a copy and permute the feature
            X_permuted = X_test_df.copy()
            X_permuted[feature] = np.random.permutation(X_permuted[feature].values)
            
            # Score with permuted feature
            permuted_score = score_func(X_permuted.values)
            
            # Calculate importance as the difference in scores
            if hasattr(model, 'predict_proba'):
                # For classification, use AUC difference
                baseline_auc = roc_auc_score(y_test, baseline_score)
                permuted_auc = roc_auc_score(y_test, permuted_score)
                importance = baseline_auc - permuted_auc
            else:
                # For autoencoder, use reconstruction error difference
                baseline_error = np.mean(baseline_score)
                permuted_error = np.mean(permuted_score)
                importance = permuted_error - baseline_error
            
            importance_scores[feature] = float(importance)
        
        # Sort features by importance
        sorted_features = sorted(importance_scores.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # Plot permutation importance
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importance (Permutation)')
        features, scores = zip(*sorted_features)
        plt.bar(range(len(scores)), [abs(s) for s in scores], align='center')
        plt.xticks(range(len(scores)), features, rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance_permutation.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        return importance_scores
    except Exception as e:
        print(f"Warning: Could not calculate permutation importance: {str(e)}")
        return {}

def generate_classification_visualizations(y_true, y_pred, y_pred_proba, output_dir='results/figures'):
    """
    Generate visualizations for classification model evaluation.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Prediction probabilities
        output_dir: Directory to save visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # ROC Curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    plt.plot(recall, precision, lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Threshold Analysis
    plt.figure(figsize=(10, 6))
    thresholds = np.linspace(0, 1, 100)
    f1_scores = []
    precision_scores = []
    recall_scores = []
    
    for threshold in thresholds:
        y_pred_t = (y_pred_proba >= threshold).astype(int)
        f1_scores.append(f1_score(y_true, y_pred_t))
        precision_scores.append(precision_score(y_true, y_pred_t))
        recall_scores.append(recall_score(y_true, y_pred_t))
    
    plt.plot(thresholds, f1_scores, label='F1 Score')
    plt.plot(thresholds, precision_scores, label='Precision')
    plt.plot(thresholds, recall_scores, label='Recall')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Threshold Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'threshold_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")

def generate_cost_sensitive_visualizations(y_true, y_pred_proba, avg_transaction_amount=500, 
                                cost_fp_ratio=0.1, cost_fn_ratio=1.0, output_dir='results/figures'):
    """
    Generate cost-sensitive visualizations for fraud detection model evaluation.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        avg_transaction_amount: Average transaction amount
        cost_fp_ratio: Cost of false positive as ratio of transaction amount
        cost_fn_ratio: Cost of false negative as ratio of transaction amount
        output_dir: Directory to save visualizations
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
    parser.add_argument('--no-mlflow', action='store_true',
                        help='Disable MLflow tracking')
    parser.add_argument('--mlflow-tracking-uri', type=str, default=None,
                        help='MLflow tracking server URI')
    args = parser.parse_args()
    
    # Set up MLflow if not disabled
    use_mlflow = not args.no_mlflow
    
    # If MLflow is enabled, try to connect to the server
    if use_mlflow:
        try:
            # Set tracking URI if provided
            if args.mlflow_tracking_uri:
                mlflow.set_tracking_uri(args.mlflow_tracking_uri)
            
            # Test connection by attempting to set experiment
            mlflow.set_experiment("Fraud_Detection_Evaluation")
            logger.info("Connected to MLflow tracking server successfully")
        except Exception as e:
            logger.warning(f"Failed to connect to MLflow server: {e}")
            logger.warning("MLflow tracking will be disabled for this evaluation.")
            logger.warning("You can run evaluation with --no-mlflow flag to avoid this warning.")
            use_mlflow = False
    
    # Load model and test data
    model = load_model(args.model_path)
    X_test, y_test, feature_names = load_test_data(args.test_data, args.model_path)
    
    # Run evaluation based on model type
    # Create a proper context manager based on MLflow availability
    if use_mlflow:
        run_context = mlflow.start_run(run_name=f"evaluate_{args.model_type}")
    else:
        # Use a dummy context manager when MLflow is disabled
        from contextlib import nullcontext
        run_context = nullcontext()
    
    with run_context:
        if args.model_type == 'classification':
            threshold = 0.5 if args.threshold is None else args.threshold
            metrics = evaluate_classification_model(
                model, X_test, y_test, 
                feature_names=feature_names,
                threshold=threshold,
                avg_transaction_amount=args.avg_transaction_amount,
                cost_fp_ratio=args.cost_fp_ratio,
                cost_fn_ratio=args.cost_fn_ratio,
                output_dir=os.path.join(args.output_dir, 'figures')
            )
            
            # Log metrics to MLflow if enabled
            if use_mlflow:
                for metric, value in metrics.items():
                    mlflow.log_metric(metric, value)
            
            # Save metrics to file
            save_metrics(metrics, args.model_type, os.path.join(args.output_dir, 'metrics'))
            
            # Log cost-related parameters if MLflow is enabled
            if use_mlflow:
                mlflow.log_param('avg_transaction_amount', args.avg_transaction_amount)
                mlflow.log_param('cost_fp_ratio', args.cost_fp_ratio)
                mlflow.log_param('cost_fn_ratio', args.cost_fn_ratio)
            
        elif args.model_type == 'autoencoder':
            try:
                metrics_result = evaluate_autoencoder_model(
                    model, X_test, y_test, 
                    feature_names=feature_names,
                    threshold=args.threshold, 
                    percentile=args.percentile,
                    optimize_threshold=True,
                    output_dir=os.path.join(args.output_dir, 'figures')
                )
                
                # Handle different return formats
                if isinstance(metrics_result, tuple) and len(metrics_result) == 2:
                    metrics, anomaly_scores = metrics_result
                else:
                    metrics = metrics_result
                    anomaly_scores = None
                
                # Log metrics to MLflow if enabled
                if use_mlflow:
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
        
        # Log figures to MLflow if they exist and MLflow is enabled
        figures_dir = os.path.join(args.output_dir, 'figures')
        
        if use_mlflow:
            if os.path.exists(figures_dir):
                try:
                    for figure_file in os.listdir(figures_dir):
                        if figure_file.endswith('.png'):
                            mlflow.log_artifact(os.path.join(figures_dir, figure_file), 'figures')
                except Exception as e:
                    print(f"Warning: Could not log figures to MLflow: {str(e)}")
            else:
                print(f"Figures directory {figures_dir} does not exist. Skipping MLflow figure logging.")
                # Create the directory for future use
                os.makedirs(figures_dir, exist_ok=True)
        elif not os.path.exists(figures_dir):
            # Create the figures directory if it doesn't exist
            print(f"Creating figures directory {figures_dir}")
            os.makedirs(figures_dir, exist_ok=True)

if __name__ == "__main__":
    main()
