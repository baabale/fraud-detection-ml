"""
Training script for fraud detection models with MLflow tracking.
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, roc_auc_score

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

# Configure logging for real-time output
os.makedirs('logs', exist_ok=True)

# Check if the root logger already has handlers to avoid duplicate logging
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/train_model.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

# Get a logger specific to this module
logger = logging.getLogger('fraud_detection_training')

# Force stdout to flush immediately
sys.stdout.reconfigure(line_buffering=True)

# Import custom modules for advanced sampling and loss functions
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.utils.advanced_sampling import get_sampling_technique
from src.utils.custom_losses import get_loss_function

# Parse command-line arguments before importing TensorFlow
# This allows us to configure GPU settings before any TensorFlow operations
def parse_args():
    parser = argparse.ArgumentParser(description='Train fraud detection models')
    parser.add_argument('--data-path', type=str,
                        help='Path to processed data')
    parser.add_argument('--model-dir', type=str,
                        help='Directory to save model artifacts')
    parser.add_argument('--run-name', type=str,
                        help='Name for the training run')
    parser.add_argument('--model-type', type=str, choices=['classification', 'autoencoder', 'both'],
                        help='Type of model to train')
    parser.add_argument('--batch-size', type=int,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int,
                        help='Number of epochs for training')
    parser.add_argument('--learning-rate', type=float,
                        help='Learning rate for optimizer')
    parser.add_argument('--dropout-rate', type=float,
                        help='Dropout rate for regularization')
    parser.add_argument('--use-mlflow', action='store_true',
                        help='Whether to use MLflow for tracking')
    parser.add_argument('--mlflow-tracking-uri', type=str,
                        help='MLflow tracking server URI')
    parser.add_argument('--experiment-name', type=str,
                        help='MLflow experiment name')
    
    # Advanced sampling arguments
    parser.add_argument('--sampling-technique', type=str, 
                        choices=['none', 'smote', 'adasyn', 'borderline_smote'],
                        default='none', help='Advanced sampling technique to use')
    parser.add_argument('--sampling-ratio', type=float, default=0.5,
                        help='Desired ratio of minority to majority class after sampling')
    parser.add_argument('--k-neighbors', type=int, default=5,
                        help='Number of nearest neighbors for sampling techniques')
    
    # Custom loss function arguments
    parser.add_argument('--loss-function', type=str,
                        choices=['binary_crossentropy', 'focal', 'weighted_focal', 
                                 'asymmetric_focal', 'adaptive_focal'],
                        default='binary_crossentropy', help='Loss function to use')
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                        help='Focusing parameter for focal loss')
    parser.add_argument('--focal-alpha', type=float, default=0.25,
                        help='Alpha parameter for focal loss')
    
    # Class weighting arguments
    parser.add_argument('--class-weight-ratio', type=float, default=None,
                        help='Weight ratio for positive class (fraud) to negative class')
    
    # Regularization arguments
    parser.add_argument('--l2-regularization', type=float, default=0.0,
                        help='L2 regularization parameter for model weights')
    
    # GPU-specific arguments
    parser.add_argument('--disable-gpu', action='store_true',
                        help='Disable GPU usage even if available')
    parser.add_argument('--single-gpu', action='store_true',
                        help='Use only a single GPU even if multiple are available')
    parser.add_argument('--batch-size-multiplier', type=int, default=2,
                        help='Multiplier for batch size when using multiple GPUs')
    parser.add_argument('--memory-growth', type=str, choices=['true', 'false'], default='true',
                        help='Enable memory growth for GPUs to prevent TensorFlow from allocating all memory')
    
    return parser.parse_args()

# Parse arguments before importing TensorFlow
args = parse_args()

# Now import TensorFlow and configure GPU settings based on arguments
import tensorflow as tf

# Configure GPU settings based on command-line arguments
strategy = None
if args.disable_gpu:
    print("GPU usage disabled by command line argument.")
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Hide all GPUs
else:
    # Check GPU availability and configure based on arguments
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            # Configure memory growth if requested
            if args.memory_growth == 'true':
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("GPU memory growth enabled.")
            
            print("\n" + "="*70)
            print(f"🚀 GPU ACCELERATION ENABLED: Using {len(gpus)} GPU(s) for training")
            print("="*70 + "\n")
            
            # Test GPU performance
            import time
            print("Testing GPU performance...")
            x = tf.random.normal([1000, 1000])
            start = time.time()
            tf.matmul(x, x)
            print(f"Matrix multiplication took: {time.time() - start:.6f} seconds")
            print(f"TensorFlow version: {tf.__version__}")
            
            # Configure multi-GPU strategy if multiple GPUs are available and not disabled
            if len(gpus) > 1 and not args.single_gpu:
                print("\n" + "="*70)
                print(f"🚀 MULTI-GPU TRAINING ENABLED: Distributing across {len(gpus)} GPUs")
                print("="*70 + "\n")
                # Create a MirroredStrategy for multi-GPU training
                strategy = tf.distribute.MirroredStrategy()
                print(f"Number of devices: {strategy.num_replicas_in_sync}")
            else:
                if args.single_gpu:
                    print("Single GPU mode enabled by command line argument.")
                else:
                    print("Single GPU detected. Using default strategy.")
        except RuntimeError as e:
            print(f"Error configuring GPUs: {e}")
    else:
        print("\n" + "="*70)
        print("⚠️ NO GPU DETECTED: Using CPU for training (this will be slower)")
        print("="*70 + "\n")

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import AUC
import mlflow
import mlflow.tensorflow
import joblib
from datetime import datetime

# Add the project root to the path to ensure imports work from any directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the configuration manager
from src.utils.config_manager import config

def load_and_preprocess_data(data_path, test_size=0.2, val_size=0.25, random_state=42, sampling_technique='none', sampling_ratio=0.5, k_neighbors=5):
    """
    Load and preprocess data for model training.
    
    Args:
        data_path (str): Path to the processed data file
        test_size (float): Proportion of data to use for testing
        val_size (float): Proportion of training data to use for validation
        random_state (int): Random seed for reproducibility
        sampling_technique (str): Advanced sampling technique to use ('none', 'smote', 'adasyn', 'borderline_smote')
        sampling_ratio (float): Desired ratio of minority to majority class after sampling
        k_neighbors (int): Number of nearest neighbors for sampling techniques
        
    Returns:
        tuple: X_train, X_val, X_test, y_train, y_val, y_test, feature_names, scaler
    """
    # Check if we're using multi-GPU
    using_multi_gpu = 'strategy' in globals() and strategy is not None
    
    # Load data with optimized settings
    print(f"Loading data from {data_path}")
    if data_path.endswith('.parquet'):
        # Use parallel reading for Parquet files
        import pyarrow.parquet as pq
        if using_multi_gpu:
            # Use multiple CPU threads for data loading to match GPU count
            num_gpus = strategy.num_replicas_in_sync if hasattr(strategy, 'num_replicas_in_sync') else 1
            num_threads = max(4, num_gpus * 2)  # Use at least 2 threads per GPU
            print(f"Using {num_threads} parallel threads for data loading")
            df = pq.read_table(data_path, use_threads=True, 
                              memory_map=True).to_pandas()
        else:
            df = pd.read_parquet(data_path)
    elif data_path.endswith('.csv'):
        if using_multi_gpu:
            # Use chunked reading for large CSV files
            num_gpus = strategy.num_replicas_in_sync if hasattr(strategy, 'num_replicas_in_sync') else 1
            chunksize = 1000000 // num_gpus  # Adjust chunk size based on GPU count
            print(f"Using chunked reading with size {chunksize} for CSV data")
            chunks = []
            for chunk in pd.read_csv(data_path, chunksize=chunksize):
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
    
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
    
    # Print column types for debugging
    print("Column dtypes:")
    print(df.dtypes)
    
    # Convert target to integer if it's boolean
    target_col = 'is_fraud'
    if df[target_col].dtype == 'bool':
        df['fraud_label'] = df[target_col].astype(int)
        target_col = 'fraud_label'
        print(f"Converted boolean target to integer: {target_col}")
    
    # Select only numeric columns for features
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Remove the target column from features if it's in numeric_cols
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    # Also remove any ID columns or other non-feature columns
    cols_to_exclude = ['transaction_id', 'sender_account', 'receiver_account', 'ip_address', 'device_hash']
    numeric_cols = [col for col in numeric_cols if col not in cols_to_exclude]
    
    print(f"Using {len(numeric_cols)} numeric features: {numeric_cols}")
    
    # Check for and handle NaN values
    nan_counts = df[numeric_cols].isna().sum()
    if nan_counts.sum() > 0:
        print(f"Warning: Found {nan_counts.sum()} NaN values across {sum(nan_counts > 0)} columns")
        print(nan_counts[nan_counts > 0])
        
        # Fill NaN values with column means
        print("Filling NaN values with column means")
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    # Separate features and target
    y = df[target_col].values
    X = df[numeric_cols].values
    
    # Double-check for any remaining NaN values
    if np.isnan(X).any():
        print("Warning: NaN values still present after filling. Replacing with zeros.")
        X = np.nan_to_num(X, nan=0.0)
    
    # Check if we have multiple classes for stratification
    unique_classes = np.unique(y)
    print(f"Unique classes in dataset: {unique_classes}")
    
    # Split into train and test sets
    if len(unique_classes) > 1:
        # Split the data into training, validation, and test sets
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size, random_state=random_state, stratify=y_train_val
        )
        
        # Apply advanced sampling techniques to the training data if specified
        if sampling_technique != 'none':
            print(f"\nApplying {sampling_technique} with sampling_ratio={sampling_ratio} and k_neighbors={k_neighbors}")
            # Get the sampling function
            sampling_func = get_sampling_technique(sampling_technique)
            if sampling_func is not None:
                try:
                    # Print class distribution before sampling
                    class_counts_before = np.bincount(y_train.astype(int))
                    print(f"Class distribution before sampling: {class_counts_before}")
                    if len(class_counts_before) > 1:
                        fraud_ratio_before = class_counts_before[1] / len(y_train) * 100
                        print(f"Fraud ratio before sampling: {fraud_ratio_before:.2f}%")
                    
                    # Apply the sampling technique
                    X_train_resampled, y_train_resampled = sampling_func(
                        X_train, y_train, 
                        sampling_strategy=sampling_ratio,
                        k_neighbors=k_neighbors,
                        random_state=random_state
                    )
                    
                    # Update the training data
                    X_train = X_train_resampled
                    y_train = y_train_resampled
                    
                    # Print class distribution after sampling
                    class_counts_after = np.bincount(y_train.astype(int))
                    print(f"Class distribution after sampling: {class_counts_after}")
                    if len(class_counts_after) > 1:
                        fraud_ratio_after = class_counts_after[1] / len(y_train) * 100
                        print(f"Fraud ratio after sampling: {fraud_ratio_after:.2f}%")
                        print(f"Sampling increased fraud ratio from {fraud_ratio_before:.2f}% to {fraud_ratio_after:.2f}%")
                except Exception as e:
                    print(f"Error applying sampling technique: {str(e)}")
                    print("Using original data without sampling.")
            else:
                print(f"Warning: Unknown sampling technique '{sampling_technique}'. Using original data.")
        else:
            print("\nNo advanced sampling technique applied.")
            
        # Print class distribution in training data
        unique, counts = np.unique(y_train, return_counts=True)
        print(f"Class distribution in training data: {dict(zip(unique, counts))}")
        print(f"Fraud ratio in training data: {np.sum(y_train == 1) / len(y_train):.4f}")
    else:
        # Use regular split if only one class (no stratification needed)
        print("Warning: Only one class present. Using regular train/test split without stratification.")
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Split training data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size, random_state=random_state
        )
    
    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    print(f"Data split: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, numeric_cols, scaler

def save_model_artifacts(model, model_type, scaler, feature_names, params, results, model_dir=None):
    """
    Save model and associated artifacts.
    
    Args:
        model: Trained model
        model_type (str): Type of model ('classification' or 'autoencoder')
        scaler: Fitted scaler
        feature_names (list): List of feature names
        params (dict): Model parameters
        results (dict): Training results
        model_dir (str, optional): Directory to save model artifacts. If None, uses the path from config.
    """
    # Use configuration if path not provided
    if model_dir is None:
        model_dir = config.get_model_path()
        
    os.makedirs(model_dir, exist_ok=True)
    print(f"Saving model artifacts to {model_dir}")
    
    # Save model using native Keras format
    model_path = os.path.join(model_dir, f"{params['run_name']}_model.keras")
    model.save(model_path, save_format='keras')
    
    # Save scaler
    scaler_path = os.path.join(model_dir, f"{params['run_name']}_scaler.joblib")
    joblib.dump(scaler, scaler_path)
    
    # Save feature names
    feature_names_path = os.path.join(model_dir, f"{params['run_name']}_feature_names.json")
    with open(feature_names_path, 'w') as f:
        json.dump(feature_names, f)
    
    # Save model parameters
    params_path = os.path.join(model_dir, f"{params['run_name']}_params.json")
    with open(params_path, 'w') as f:
        json.dump(params, f)
    
    # Save training results
    results_path = os.path.join(model_dir, f"{params['run_name']}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f)

def evaluate_classification_model(model, X_test, y_test):
    """
    Evaluate a classification model and return metrics.
    
    Args:
        model: Trained classification model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Predictions
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Check for class distribution in test data
    unique_classes = np.unique(y_test)
    print(f"Unique classes in test data: {unique_classes}")
    
    # Calculate metrics
    metrics = {
        'accuracy': np.mean(y_pred == y_test),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
    }
    
    # Only calculate AUC if we have both classes and no NaN values
    if len(unique_classes) > 1:
        # Check for NaN values in predictions
        if np.isnan(y_pred_proba).any():
            print("Warning: NaN values found in predictions. Replacing with 0.")
            y_pred_proba = np.nan_to_num(y_pred_proba, nan=0.0)
        
        try:
            metrics['auc'] = roc_auc_score(y_test, y_pred_proba)
        except Exception as e:
            print(f"Warning: Could not calculate AUC: {str(e)}")
            metrics['auc'] = 0.0
    else:
        print("Warning: Only one class present in test data. AUC cannot be calculated.")
        metrics['auc'] = 0.0  # Default value when AUC can't be calculated
    
    try:
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        for label, scores in report.items():
            if isinstance(scores, dict):
                for metric, value in scores.items():
                    metrics[f'{label}_{metric}'] = value
    except Exception as e:
        print(f"Warning: Could not generate classification report: {str(e)}")
    
    return metrics

def compute_anomaly_scores(model, X):
    """
    Compute anomaly scores for autoencoder model.
    
    Args:
        model: Trained autoencoder model
        X: Input features
        
    Returns:
        numpy.ndarray: Array of anomaly scores
    """
    # Get reconstructions
    X_pred = model.predict(X)
    
    # Compute reconstruction error (MSE) for each sample
    mse = np.mean(np.square(X - X_pred), axis=1)
    
    return mse

def evaluate_autoencoder_model(model, X_test, y_test, threshold=None):
    """
    Evaluate an autoencoder model for anomaly detection.
    
    Args:
        model: Trained autoencoder model
        X_test: Test features
        y_test: Test labels
        threshold: Anomaly score threshold (if None, determined from data)
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Compute anomaly scores
    anomaly_scores = compute_anomaly_scores(model, X_test)
    
    # Determine threshold if not provided
    if threshold is None:
        # Use 95th percentile of non-fraud scores as threshold
        non_fraud_indices = (y_test == 0)
        non_fraud_scores = anomaly_scores[non_fraud_indices]
        threshold = np.percentile(non_fraud_scores, 95)
    
    # Classify based on threshold
    y_pred = (anomaly_scores > threshold).astype(int)
    
    # Calculate metrics
    metrics = {
        'accuracy': np.mean(y_pred == y_test),
        'threshold': threshold,
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
    }
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    for label, scores in report.items():
        if isinstance(scores, dict):
            for metric, value in scores.items():
                metrics[f'{label}_{metric}'] = value
    
    return metrics, anomaly_scores

def get_loss_function_for_model(loss_function_name, focal_gamma=2.0, focal_alpha=0.25, class_weights=None):
    """
    Get the appropriate loss function based on name.
    
    Args:
        loss_function_name (str): Name of the loss function
        focal_gamma (float): Focusing parameter for focal loss
        focal_alpha (float): Alpha parameter for focal loss
        class_weights (dict): Class weights for weighted loss functions
        
    Returns:
        Loss function to use in model compilation
    """
    if loss_function_name == 'binary_crossentropy':
        return 'binary_crossentropy'
    
    # Get custom loss function
    try:
        loss_params = {
            'gamma': focal_gamma,
            'alpha': focal_alpha
        }
        
        # For weighted_focal loss, add class weights
        if loss_function_name == 'weighted_focal' and class_weights is not None:
            loss_params['class_weights'] = class_weights
            
        loss = get_loss_function(loss_function_name, **loss_params)
        print(f"Using custom loss function: {loss_function_name} with parameters: {loss_params}")
        return loss
    except Exception as e:
        print(f"Warning: Error configuring custom loss function: {str(e)}")
        print("Falling back to binary_crossentropy")
        return 'binary_crossentropy'


def train_classification_model(X_train, y_train, X_val, y_val, input_dim, hidden_layers=[64, 32], learning_rate=0.001, batch_size=64, epochs=50, class_weight=None, model_path=None, loss_function='binary_crossentropy', focal_gamma=2.0, focal_alpha=0.25, l2_regularization=0.0):
    """
    Train a classification model for fraud detection.
    
    Args:
        X_train (array): Training features
        y_train (array): Training labels
        X_val (array): Validation features
        y_val (array): Validation labels
        input_dim (int): Input dimension
        hidden_layers (list): List of hidden layer sizes
        learning_rate (float): Learning rate for optimizer
        batch_size (int): Batch size for training
        epochs (int): Number of training epochs
        class_weight (dict, optional): Class weights for imbalanced data
        model_path (str, optional): Path to save the trained model
        loss_function (str): Loss function to use ('binary_crossentropy', 'focal', 'weighted_focal', etc.)
        focal_gamma (float): Focusing parameter for focal loss
        focal_alpha (float): Alpha parameter for focal loss
        
    Returns:
        model: Trained classification model
    """
    # Use multi-GPU strategy if available
    if 'strategy' in globals() and strategy is not None:
        print(f"Creating classification model with multi-GPU strategy")
        
    # Get the appropriate loss function
    loss = get_loss_function_for_model(
        loss_function_name=loss_function,
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha,
        class_weights=class_weight
    )
    
    # Define metrics
    metrics = ['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    
    if strategy is not None and hasattr(strategy, 'scope'):
        # Multi-GPU model creation
        with strategy.scope():
            # Create model
            input_layer = Input(shape=(input_dim,))
            x = input_layer
            
            # Add hidden layers
            for units in hidden_layers:
                x = Dense(units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_regularization))(x)
                x = Dropout(0.4)(x)
            
            # Output layer
            output_layer = Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(l2_regularization))(x)
            
            # Create model
            model = Model(inputs=input_layer, outputs=output_layer)
            
            # Compile model
            optimizer = Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    else:
        # Single-GPU or CPU model creation
        # Create model
        model = Sequential()
        model.add(Input(shape=(input_dim,)))
        
        # Add hidden layers
        for units in hidden_layers:
            model.add(Dense(units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_regularization)))
            model.add(Dropout(0.4))
        
        # Output layer (binary classification)
        model.add(Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(l2_regularization)))
        
        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ]
    
    # Add model checkpoint if path is provided
    if model_path:
        callbacks.append(ModelCheckpoint(model_path, save_best_only=True))
    
    # Create optimized TensorFlow datasets for training
    if 'strategy' in globals() and strategy is not None and hasattr(strategy, 'num_replicas_in_sync'):
        # Optimize batch size for multi-GPU training
        # Increase batch size proportionally to the number of GPUs
        effective_batch_size = batch_size * strategy.num_replicas_in_sync
        print(f"Using effective batch size of {effective_batch_size} for {strategy.num_replicas_in_sync} GPUs")
        
        # Create TensorFlow datasets with prefetching and caching
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.cache()
        train_dataset = train_dataset.shuffle(buffer_size=10000)
        train_dataset = train_dataset.batch(effective_batch_size)
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_dataset = val_dataset.batch(effective_batch_size)
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
        
        # Train model with optimized datasets
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1
        )
    else:
        # Standard training for single GPU
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1
        )
    
    return model, history

def run_classification_experiment(X_train, X_val, X_test, y_train, y_val, y_test, 
                                 input_dim, params, model_dir, feature_names=None):    
    """
    Run a classification model experiment with MLflow tracking.
    
    Args:
        X_train, X_val, X_test: Training, validation, and test features
        y_train, y_val, y_test: Training, validation, and test labels
        input_dim: Input dimension
        params: Model and training parameters
        model_dir: Directory to save the model
        feature_names: List of feature names
        
    Returns:
        dict: Evaluation metrics
    """
    # Import visualization libraries here to avoid issues if running in headless mode
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set up MLflow experiment
    mlflow.set_experiment(params['experiment_name'])
    
    with mlflow.start_run(run_name=params['run_name']):
        # Log parameters
        log_params = {
            'model_type': 'classification',
            'batch_size': params['batch_size'],
            'epochs': params['epochs'],
            'learning_rate': params.get('learning_rate', 0.001),
            'input_dim': input_dim,
            'sampling_technique': params.get('sampling_technique', 'none'),
            'loss_function': params.get('loss_function', 'binary_crossentropy'),
            'l2_regularization': params.get('l2_regularization', 0.0)
        }
        
        # Log additional parameters for advanced sampling if used
        if params.get('sampling_technique', 'none') != 'none':
            log_params['sampling_ratio'] = params.get('sampling_ratio', 0.5)
            log_params['k_neighbors'] = params.get('k_neighbors', 5)
            
        # Log additional parameters for custom loss functions if used
        if params.get('loss_function', 'binary_crossentropy') != 'binary_crossentropy':
            log_params['focal_gamma'] = params.get('focal_gamma', 2.0)
            log_params['focal_alpha'] = params.get('focal_alpha', 0.25)
            
        mlflow.log_params(log_params)
        
        # Set up class weights if not provided
        if 'class_weight' not in params or params['class_weight'] is None:
            # Calculate class weights based on class distribution
            # This helps with imbalanced datasets (fraud detection)
            class_counts = np.bincount(y_train.astype(int))
            total_samples = len(y_train)
            
            # Check if class_weight_ratio is provided
            if 'class_weight_ratio' in params and params['class_weight_ratio'] is not None:
                # Use the specified ratio for positive class (fraud)
                ratio = params['class_weight_ratio']
                print(f"Using specified class weight ratio: {ratio}")
                class_weights = {0: 1.0, 1: ratio}
            else:
                # Calculate balanced class weights
                class_weights = {i: total_samples / (len(class_counts) * count) for i, count in enumerate(class_counts)}
            
            print(f"Using class weights: {class_weights}")
        else:
            class_weights = params.get('class_weight')
            
        mlflow.log_param('class_weight', str(class_weights))
        
        # Model path
        model_path = os.path.join(model_dir, f"{params['run_name']}_model.h5")
        
        # Get loss function parameters
        loss_function = params.get('loss_function', 'binary_crossentropy')
        focal_gamma = params.get('focal_gamma', 2.0)
        focal_alpha = params.get('focal_alpha', 0.25)
        
        # Log loss function parameters
        mlflow.log_param('loss_function', loss_function)
        if loss_function != 'binary_crossentropy':
            mlflow.log_param('focal_gamma', focal_gamma)
            mlflow.log_param('focal_alpha', focal_alpha)
        
        # Train the model
        model, history = train_classification_model(
            X_train, y_train, X_val, y_val,
            input_dim=input_dim,
            hidden_layers=params.get('hidden_layers', [64, 32]),
            learning_rate=params.get('learning_rate', 0.001),
            batch_size=params.get('batch_size', 64),
            epochs=params.get('epochs', 20),
            class_weight=class_weights,
            model_path=model_path,
            loss_function=loss_function,
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha,
            l2_regularization=params.get('l2_regularization', 0.0)
        )
        
        # Log training metrics
        for epoch, metrics in enumerate(history.history.items()):
            metric_name, values = metrics
            for i, value in enumerate(values):
                mlflow.log_metric(f"train_{metric_name}", value, step=i)
        
        # Evaluate model
        metrics = evaluate_classification_model(model, X_test, y_test)
        
        # Log evaluation metrics
        for metric_name, value in metrics.items():
            if not isinstance(value, list):
                mlflow.log_metric(metric_name, value)
        
        # Log confusion matrix as a figure
        cm = np.array(metrics['confusion_matrix'])
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)
        
        # Create model signature
        from mlflow.models.signature import infer_signature

        # Generate a sample input for signature
        signature = infer_signature(X_test[:1], model.predict(X_test[:1]))

        # Save model with proper file extension before logging to MLflow
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "model.keras")
            model.save(model_path, save_format="keras")
            
            # Log model with signature and sample input
            mlflow.tensorflow.log_model(
                model, 
                "classification_model",
                signature=signature,
                input_example=X_test[:1]
            )

        # Log feature names
        mlflow.log_dict(feature_names, "feature_names.json")
        
        return model, metrics

def train_autoencoder_model(X_train, X_val, input_dim, batch_size=256, epochs=20,
                          learning_rate=0.001, hidden_layers=None, encoding_dim=16,
                          model_path=None):
    """
    Train an autoencoder model for anomaly detection.
    
    Args:
        X_train: Training features
        X_val: Validation features
        input_dim: Input dimension
        batch_size: Batch size for training
        epochs: Number of epochs
        learning_rate: Learning rate for optimizer
        hidden_layers: List of hidden layer sizes
        encoding_dim: Dimension of the encoding layer
        model_path: Path to save the model
        
    Returns:
        tuple: Trained model and training history
    """
    # Set default hidden layers if not provided
    if hidden_layers is None:
        hidden_layers = [64, 32]
    
    # Use multi-GPU strategy if available
    if 'strategy' in globals() and strategy is not None:
        print(f"Creating autoencoder model with multi-GPU strategy")
        with strategy.scope():
            # Build encoder
            input_layer = Input(shape=(input_dim,))
            encoded = Dense(hidden_layers[0], activation='relu')(input_layer)
            
            # Add encoding layers
            for units in hidden_layers[1:]:
                encoded = Dense(units, activation='relu')(encoded)
            
            # Bottleneck layer
            bottleneck = Dense(encoding_dim, activation='relu')(encoded)
            
            # Build decoder (mirror of encoder)
            decoded = Dense(hidden_layers[-1], activation='relu')(bottleneck)
            
            # Add decoding layers
            for units in reversed(hidden_layers[:-1]):
                decoded = Dense(units, activation='relu')(decoded)
            
            # Output layer
            output_layer = Dense(input_dim, activation='linear')(decoded)
            
            # Create model
            model = Model(inputs=input_layer, outputs=output_layer)
            
            # Compile model
            optimizer = Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer, loss='mse')
    else:
        # Single-GPU or CPU model creation
        # Build encoder
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(hidden_layers[0], activation='relu')(input_layer)
        
        # Add encoding layers
        for units in hidden_layers[1:]:
            encoded = Dense(units, activation='relu')(encoded)
        
        # Bottleneck layer
        bottleneck = Dense(encoding_dim, activation='relu')(encoded)
        
        # Build decoder (mirror of encoder)
        decoded = Dense(hidden_layers[-1], activation='relu')(bottleneck)
        
        # Add decoding layers
        for units in reversed(hidden_layers[:-1]):
            decoded = Dense(units, activation='relu')(decoded)
        
        # Output layer
        output_layer = Dense(input_dim, activation='linear')(decoded)
        
        # Create model
        model = Model(inputs=input_layer, outputs=output_layer)
        
        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse')
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ]
    
    # Add model checkpoint if path is provided
    if model_path:
        callbacks.append(ModelCheckpoint(model_path, save_best_only=True))
    
    # Create optimized TensorFlow datasets for training
    if 'strategy' in globals() and strategy is not None and hasattr(strategy, 'num_replicas_in_sync'):
        # Optimize batch size for multi-GPU training
        # Increase batch size proportionally to the number of GPUs
        effective_batch_size = batch_size * strategy.num_replicas_in_sync
        print(f"Using effective batch size of {effective_batch_size} for {strategy.num_replicas_in_sync} GPUs")
        
        # Create TensorFlow datasets with prefetching and caching
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, X_train))  # Autoencoder trains to reconstruct input
        train_dataset = train_dataset.cache()
        train_dataset = train_dataset.shuffle(buffer_size=10000)
        train_dataset = train_dataset.batch(effective_batch_size)
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, X_val))  # Same for validation
        val_dataset = val_dataset.batch(effective_batch_size)
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
        
        # Train model with optimized datasets
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
    else:
        # Standard training for single GPU
        history = model.fit(
            X_train, X_train,  # Autoencoder trains to reconstruct input
            validation_data=(X_val, X_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
    
    return model, history

def run_autoencoder_experiment(X_train, X_val, X_test, y_train, y_val, y_test, 
                              input_dim, params, model_dir, feature_names=None):
    """
    Run an autoencoder model experiment with MLflow tracking.
    
    Args:
        X_train, X_val, X_test: Training, validation, and test features
        y_train, y_val, y_test: Training, validation, and test labels
        input_dim: Input dimension
        params: Model and training parameters
        model_dir: Directory to save the model
        feature_names: List of feature names
        
    Returns:
        dict: Evaluation metrics
    """
    # Import visualization libraries here to avoid issues if running in headless mode
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set up MLflow experiment
    mlflow.set_experiment(params['experiment_name'])
    
    with mlflow.start_run(run_name=params['run_name']):
        # Log parameters
        mlflow.log_params({
            'model_type': 'autoencoder',
            'hidden_layers': str(params['hidden_layers']),
            'encoding_dim': params['encoding_dim'],
            'batch_size': params['batch_size'],
            'epochs': params['epochs'],
            'input_dim': input_dim
        })
        
        # Filter training data to include only non-fraud transactions
        X_train_normal = X_train[y_train == 0]
        if X_val is not None and y_val is not None:
            X_val_normal = X_val[y_val == 0]
        else:
            X_val_normal = None
        
        # Model path
        model_path = os.path.join(model_dir, f"{params['run_name']}_model.h5")
        
        # Train model
        model, history = train_autoencoder_model(
            X_train_normal, X_val_normal,
            input_dim=input_dim,
            batch_size=params['batch_size'],
            epochs=params['epochs'],
            model_path=model_path
        )
        
        # Log training metrics
        for epoch, metrics in enumerate(history.history.items()):
            metric_name, values = metrics
            for i, value in enumerate(values):
                mlflow.log_metric(f"train_{metric_name}", value, step=i)
        
        # Evaluate model
        metrics, anomaly_scores = evaluate_autoencoder_model(model, X_test, y_test)
        
        # Log evaluation metrics
        for metric_name, value in metrics.items():
            if not isinstance(value, list):
                mlflow.log_metric(metric_name, value)
        
        # Log confusion matrix as a figure
        cm = np.array(metrics['confusion_matrix'])
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)
        
        # Log anomaly score distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(anomaly_scores[y_test == 0], label='Normal', alpha=0.5, ax=ax)
        sns.histplot(anomaly_scores[y_test == 1], label='Fraud', alpha=0.5, ax=ax)
        ax.axvline(metrics['threshold'], color='red', linestyle='--', label='Threshold')
        ax.set_xlabel('Anomaly Score')
        ax.set_ylabel('Count')
        ax.set_title('Anomaly Score Distribution')
        ax.legend()
        mlflow.log_figure(fig, "anomaly_scores.png")
        plt.close(fig)
        
        # Log model
        # Create model signature
        from mlflow.models.signature import infer_signature
        
        # Generate a sample input for signature
        signature = infer_signature(X_test[:1], model.predict(X_test[:1]))
        
        # Save model with proper file extension before logging to MLflow
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "model.keras")
            model.save(model_path, save_format="keras")
            
            # Log model with signature and sample input
            mlflow.tensorflow.log_model(
                model, 
                "autoencoder_model",
                signature=signature,
                input_example=X_test[:1]
            )

        # Log feature names
        mlflow.log_dict(feature_names, "feature_names.json")

        # Log threshold value
        if 'threshold' in metrics:
            mlflow.log_dict({"threshold": float(metrics['threshold'])}, "threshold.json")        
        return model, metrics

def main():
    """Main function to train fraud detection models."""
    # Handle GPU configuration based on command-line arguments
    parser = argparse.ArgumentParser(description='Train fraud detection models')
    parser.add_argument('--data-path', type=str,
                        help='Path to processed data')
    parser.add_argument('--model-dir', type=str,
                        help='Directory to save model artifacts')
    parser.add_argument('--run-name', type=str,
                        help='Name for the training run')
    parser.add_argument('--model-type', type=str, choices=['classification', 'autoencoder', 'both'],
                        help='Type of model to train')
    parser.add_argument('--batch-size', type=int,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int,
                        help='Number of epochs for training')
    parser.add_argument('--learning-rate', type=float,
                        help='Learning rate for optimizer')
    parser.add_argument('--dropout-rate', type=float,
                        help='Dropout rate for regularization')
    parser.add_argument('--use-mlflow', action='store_true',
                        help='Whether to use MLflow for tracking')
    parser.add_argument('--mlflow-tracking-uri', type=str,
                        help='MLflow tracking server URI')
    parser.add_argument('--experiment-name', type=str,
                        help='MLflow experiment name')
    
    # Advanced sampling arguments
    parser.add_argument('--sampling-technique', type=str, 
                        choices=['none', 'smote', 'adasyn', 'borderline_smote'],
                        help='Advanced sampling technique to use')
    parser.add_argument('--sampling-ratio', type=float,
                        help='Desired ratio of minority to majority class after sampling')
    parser.add_argument('--k-neighbors', type=int,
                        help='Number of nearest neighbors for sampling techniques')
    
    # Custom loss function arguments
    parser.add_argument('--loss-function', type=str,
                        choices=['binary_crossentropy', 'focal', 'weighted_focal', 
                                 'asymmetric_focal', 'adaptive_focal'],
                        help='Loss function to use')
    parser.add_argument('--focal-gamma', type=float,
                        help='Focusing parameter for focal loss')
    parser.add_argument('--focal-alpha', type=float,
                        help='Alpha parameter for focal loss')
    parser.add_argument('--class-weight-ratio', type=float,
                        help='Weight ratio for positive class (fraud) to negative class')
    parser.add_argument('--l2-regularization', type=float,
                        help='L2 regularization strength for model weights')
    
    # GPU-specific arguments
    parser.add_argument('--disable-gpu', action='store_true',
                        help='Disable GPU usage even if available')
    parser.add_argument('--single-gpu', action='store_true',
                        help='Use only a single GPU even if multiple are available')
    parser.add_argument('--batch-size-multiplier', type=int, default=2,
                        help='Multiplier for batch size when using multiple GPUs')
    parser.add_argument('--memory-growth', type=str, choices=['true', 'false'], default='true',
                        help='Enable memory growth for GPUs to prevent TensorFlow from allocating all memory')
    
    args = parser.parse_args()
    
    if args.disable_gpu:
        print("GPU usage disabled by command line argument.")
        tf.config.set_visible_devices([], 'GPU')
        strategy = None
    else:
        # Check GPU availability and configure based on arguments
        gpus = tf.config.list_physical_devices('GPU')
        
        # Configure memory growth if requested
        if args.memory_growth == 'true' and gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("GPU memory growth enabled.")
            except RuntimeError as e:
                print(f"Error configuring GPU memory growth: {e}")
        
        # Configure multi-GPU strategy by default if multiple GPUs are available
        # Only use single GPU if explicitly requested
        if len(gpus) > 1 and not args.single_gpu:
            try:
                print(f"\n" + "="*70)
                print(f"🚀 MULTI-GPU TRAINING ENABLED: Distributing across {len(gpus)} GPUs")
                print("="*70 + "\n")
                
                # Create a MirroredStrategy for multi-GPU training
                strategy = tf.distribute.MirroredStrategy()
                print(f"Number of devices in sync: {strategy.num_replicas_in_sync}")
                
                # Adjust batch size based on multiplier
                if args.batch_size:
                    args.batch_size *= args.batch_size_multiplier
                    print(f"Adjusted batch size for multi-GPU: {args.batch_size}")
            except Exception as e:
                print(f"Error configuring multi-GPU strategy: {e}")
                strategy = None
        else:
            if len(gpus) > 1 and args.single_gpu:
                print(f"Multiple GPUs detected but using only one GPU due to --single-gpu flag.")
            elif gpus:
                print(f"Using {len(gpus)} GPU(s) for model training.")
            else:
                print("No GPU available. Using CPU for model training.")
            strategy = None
    
    # Use configuration values if arguments are not provided
    data_path = args.data_path or config.get_data_path('processed_path')
    model_dir = args.model_dir or config.get_model_path()
    run_name = args.run_name or f'fraud_detection_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    model_type = args.model_type or config.get('models.model_type', 'both')
    
    # Get model parameters from config if not provided
    classification_params = config.get_model_params('classification')
    autoencoder_params = config.get_model_params('autoencoder')
    
    batch_size = args.batch_size or classification_params.get('batch_size', 256)
    epochs = args.epochs or classification_params.get('epochs', 20)
    learning_rate = args.learning_rate or config.get('models.learning_rate', 0.001)
    dropout_rate = args.dropout_rate or classification_params.get('dropout_rate', 0.4)
    
    # Advanced sampling parameters
    sampling_technique = args.sampling_technique or 'none'
    sampling_ratio = args.sampling_ratio or 0.5
    k_neighbors = args.k_neighbors or 5
    
    # Custom loss function parameters
    loss_function = args.loss_function or 'binary_crossentropy'
    focal_gamma = args.focal_gamma or 2.0
    focal_alpha = args.focal_alpha or 0.25
    
    # Class weighting parameters
    class_weight_ratio = args.class_weight_ratio
    
    # Regularization parameters
    l2_regularization = args.l2_regularization or 0.0
    
    # MLflow settings
    mlflow_config = config.get_mlflow_config()
    use_mlflow = args.use_mlflow or config.get('mlflow.enabled', False)
    mlflow_tracking_uri = args.mlflow_tracking_uri or mlflow_config.get('tracking_uri', 'http://localhost:5000')
    experiment_name = args.experiment_name or mlflow_config.get('experiment_name', 'Fraud_Detection_Experiment')
    
    # Load and prepare data
    print(f"\nUsing sampling technique: {sampling_technique}")
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names, scaler = load_and_preprocess_data(
        data_path, 
        sampling_technique=sampling_technique,
        sampling_ratio=sampling_ratio,
        k_neighbors=k_neighbors
    )
    input_dim = X_train.shape[1]
    
    # Run experiments
    if model_type in ['classification', 'both']:
        # Classification model parameters
        classification_params = {
            'experiment_name': experiment_name,
            'run_name': run_name,
            'hidden_layers': classification_params.get('hidden_layers', [64, 32]),
            'dropout_rate': dropout_rate,
            'batch_size': batch_size,
            'epochs': epochs,
            'learning_rate': learning_rate,
            'model_type': 'classification',
            'use_mlflow': use_mlflow,
            'loss_function': loss_function,
            'focal_gamma': focal_gamma,
            'focal_alpha': focal_alpha,
            'class_weight_ratio': class_weight_ratio,
            'sampling_technique': sampling_technique,
            'l2_regularization': l2_regularization,
            'mlflow_tracking_uri': mlflow_tracking_uri
        }
        
        # Run classification experiment
        print("Training classification model...")
        # For classification model
        classification_model, classification_metrics = run_classification_experiment(
            X_train, X_val, X_test, y_train, y_val, y_test,
            input_dim, classification_params, model_dir, feature_names
        )
        
        print("Classification model results:")
        for metric, value in classification_metrics.items():
            if not isinstance(value, list):
                print(f"  {metric}: {value}")
        
        # Save model artifacts
        save_model_artifacts(
            classification_model,
            'classification',
            scaler,
            feature_names,
            classification_params,
            classification_metrics,
            model_dir
        )
    
    if model_type in ['autoencoder', 'both']:
        # Autoencoder model parameters
        autoencoder_params = {
            'experiment_name': experiment_name,
            'run_name': run_name,
            'hidden_layers': autoencoder_params.get('hidden_layers', [64, 32]),
            'encoding_dim': autoencoder_params.get('encoding_dim', 16),
            'batch_size': batch_size,
            'epochs': epochs,
            'learning_rate': learning_rate,
            'use_mlflow': use_mlflow,
            'mlflow_tracking_uri': mlflow_tracking_uri
        }
        
        # Run autoencoder experiment
        print("Training autoencoder model...")
        # For autoencoder model
        autoencoder_model, autoencoder_metrics = run_autoencoder_experiment(
            X_train, X_val, X_test, y_train, y_val, y_test,
            input_dim, autoencoder_params, model_dir, feature_names
        )
        
        print("Autoencoder model results:")
        for metric, value in autoencoder_metrics.items():
            if not isinstance(value, list):
                print(f"  {metric}: {value}")
                
        # Save autoencoder model artifacts
        save_model_artifacts(
            autoencoder_model,
            'autoencoder',
            scaler,
            feature_names,
            autoencoder_params,
            autoencoder_metrics,
            model_dir
        )

if __name__ == "__main__":
    main()
