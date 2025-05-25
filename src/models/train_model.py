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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
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

def load_data(data_path=None):
    """
    Load and prepare data for model training.
    
    Args:
        data_path (str, optional): Path to the processed data file. If None, uses the path from config.
        
    Returns:
        tuple: X_train, X_val, X_test, y_train, y_val, y_test, feature_names, scaler
    """
    # Use configuration if path not provided
    if data_path is None:
        data_path = config.get_data_path('processed_path')
        
    print(f"Loading data from {data_path}")
    # For Parquet files
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    # For CSV files
    elif data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
    
    print(f"Loaded {len(df)} transactions from {data_path}")
    
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
        # Use stratified split if multiple classes
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Split training data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
        )
    else:
        # Use regular split if only one class (no stratification needed)
        print("Warning: Only one class present. Using regular train/test split without stratification.")
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Split training data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.25, random_state=42
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

def train_classification_model(X_train, y_train, X_val, y_val, input_dim, batch_size=256, 
                           epochs=20, learning_rate=0.001, dropout_rate=0.4, 
                           hidden_layers=None, class_weight=None, model_path=None):
    """
    Train a classification model for fraud detection.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        input_dim: Input dimension
        batch_size: Batch size for training
        epochs: Number of epochs
        learning_rate: Learning rate for optimizer
        dropout_rate: Dropout rate for regularization
        hidden_layers: List of hidden layer sizes
        class_weight: Class weights for imbalanced data
        model_path: Path to save the model
        
    Returns:
        tuple: Trained model and training history
    """
    # Set default hidden layers if not provided
    if hidden_layers is None:
        hidden_layers = [64, 32]
    
    # Build model
    model = Sequential()
    
    # Input layer
    model.add(Dense(hidden_layers[0], activation='relu', input_dim=input_dim))
    model.add(Dropout(dropout_rate))
    
    # Hidden layers
    for units in hidden_layers[1:]:
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(dropout_rate))
    
    # Output layer
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ]
    
    # Add model checkpoint if path is provided
    if model_path:
        callbacks.append(ModelCheckpoint(model_path, save_best_only=True))
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
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
        mlflow.log_params({
            'model_type': 'classification',
            'hidden_layers': str(params['hidden_layers']),
            'dropout_rate': params['dropout_rate'],
            'batch_size': params['batch_size'],
            'epochs': params['epochs'],
            'input_dim': input_dim
        })
        
        # Handle class imbalance
        class_counts = np.bincount(y_train.astype(int))
        print(f"Class distribution in training data: {class_counts}")
        
        # Check if we have both classes (fraud and non-fraud)
        if len(class_counts) > 1 and class_counts[1] > 0:
            class_weight = {0: 1.0, 1: class_counts[0] / class_counts[1]}
        else:
            # If only one class is present, use balanced weights
            print("Warning: Only one class present in training data. Using balanced weights.")
            class_weight = {0: 1.0}
            
        mlflow.log_param('class_weight', str(class_weight))
        
        # Model path
        model_path = os.path.join(model_dir, f"{params['run_name']}_model.h5")
        
        # Train model
        model, history = train_classification_model(
            X_train, y_train, X_val, y_val,
            input_dim=input_dim,
            batch_size=params['batch_size'],
            epochs=params['epochs'],
            class_weight=class_weight,
            model_path=model_path
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
    
    # Train model
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
    """
    Main function to train fraud detection models.
    """
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
    args = parser.parse_args()
    
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
    
    # MLflow settings
    mlflow_config = config.get_mlflow_config()
    use_mlflow = args.use_mlflow or config.get('mlflow.enabled', False)
    mlflow_tracking_uri = args.mlflow_tracking_uri or mlflow_config.get('tracking_uri', 'http://localhost:5000')
    experiment_name = args.experiment_name or mlflow_config.get('experiment_name', 'Fraud_Detection_Experiment')
    
    # Load and prepare data
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names, scaler = load_data(data_path)
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
            'use_mlflow': use_mlflow,
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
