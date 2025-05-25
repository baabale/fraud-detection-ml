"""
Training script for fraud detection models with MLflow tracking.
"""
import os
import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.tensorflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import tensorflow as tf

# Import local modules
from fraud_model import (
    create_classification_model, 
    train_classification_model,
    create_autoencoder_model,
    train_autoencoder_model,
    compute_anomaly_scores
)

def load_processed_data(data_path):
    """
    Load processed transaction data.
    
    Args:
        data_path (str): Path to the processed data file
        
    Returns:
        DataFrame: Pandas DataFrame containing the processed data
    """
    # For Parquet files
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    # For CSV files
    elif data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
    
    print(f"Loaded {len(df)} transactions from {data_path}")
    return df

def prepare_data(df, target_col='is_fraud', test_size=0.2, val_size=0.25):
    """
    Prepare data for model training by splitting and scaling.
    
    Args:
        df (DataFrame): Processed transaction data
        target_col (str): Name of the target column
        test_size (float): Proportion of data to use for testing
        val_size (float): Proportion of training data to use for validation
        
    Returns:
        tuple: X_train, X_val, X_test, y_train, y_val, y_test, scaler
    """
    # Separate features and target
    y = df[target_col].values
    X = df.drop(columns=[target_col]).values
    
    # Split into train and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Split training data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, random_state=42, stratify=y_train_val
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    print(f"Data split: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler

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
    
    # Calculate metrics
    metrics = {
        'accuracy': np.mean(y_pred == y_test),
        'auc': roc_auc_score(y_test, y_pred_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
    }
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    for label, scores in report.items():
        if isinstance(scores, dict):
            for metric, value in scores.items():
                metrics[f'{label}_{metric}'] = value
    
    return metrics

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

def run_classification_experiment(X_train, X_val, X_test, y_train, y_val, y_test, 
                                 input_dim, params, model_dir):
    """
    Run a classification model experiment with MLflow tracking.
    
    Args:
        X_train, X_val, X_test: Training, validation, and test features
        y_train, y_val, y_test: Training, validation, and test labels
        input_dim: Input dimension
        params: Model and training parameters
        model_dir: Directory to save the model
        
    Returns:
        dict: Evaluation metrics
    """
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
        class_weight = {0: 1.0, 1: class_counts[0] / class_counts[1]}
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
        
        # Log model
        mlflow.tensorflow.log_model(model, "model")
        
        return metrics

def run_autoencoder_experiment(X_train, X_val, X_test, y_train, y_val, y_test, 
                              input_dim, params, model_dir):
    """
    Run an autoencoder model experiment with MLflow tracking.
    
    Args:
        X_train, X_val, X_test: Training, validation, and test features
        y_train, y_val, y_test: Training, validation, and test labels
        input_dim: Input dimension
        params: Model and training parameters
        model_dir: Directory to save the model
        
    Returns:
        dict: Evaluation metrics
    """
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
        mlflow.tensorflow.log_model(model, "model")
        
        return metrics

def main():
    """
    Main function to run the training pipeline.
    """
    parser = argparse.ArgumentParser(description='Train fraud detection models')
    parser.add_argument('--data-path', type=str, default='../../data/processed/transactions.parquet',
                        help='Path to processed data file')
    parser.add_argument('--model-type', type=str, choices=['classification', 'autoencoder', 'both'],
                        default='both', help='Type of model to train')
    parser.add_argument('--experiment-name', type=str, default='Fraud_Detection_Experiment',
                        help='MLflow experiment name')
    parser.add_argument('--model-dir', type=str, default='../../results/models',
                        help='Directory to save trained models')
    args = parser.parse_args()
    
    # Create model directory if it doesn't exist
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Load and prepare data
    df = load_processed_data(args.data_path)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data(df)
    input_dim = X_train.shape[1]
    
    # Import visualization libraries here to avoid issues if running in headless mode
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Run experiments
    if args.model_type in ['classification', 'both']:
        # Classification model parameters
        classification_params = {
            'experiment_name': args.experiment_name,
            'run_name': 'classification_model',
            'hidden_layers': [128, 64, 32],
            'dropout_rate': 0.4,
            'batch_size': 256,
            'epochs': 20
        }
        
        # Run classification experiment
        print("Training classification model...")
        classification_metrics = run_classification_experiment(
            X_train, X_val, X_test, y_train, y_val, y_test,
            input_dim, classification_params, args.model_dir
        )
        print("Classification model results:")
        for metric, value in classification_metrics.items():
            if not isinstance(value, list):
                print(f"  {metric}: {value}")
    
    if args.model_type in ['autoencoder', 'both']:
        # Autoencoder model parameters
        autoencoder_params = {
            'experiment_name': args.experiment_name,
            'run_name': 'autoencoder_model',
            'hidden_layers': [64, 32],
            'encoding_dim': 16,
            'batch_size': 256,
            'epochs': 30
        }
        
        # Run autoencoder experiment
        print("Training autoencoder model...")
        autoencoder_metrics = run_autoencoder_experiment(
            X_train, X_val, X_test, y_train, y_val, y_test,
            input_dim, autoencoder_params, args.model_dir
        )
        print("Autoencoder model results:")
        for metric, value in autoencoder_metrics.items():
            if not isinstance(value, list):
                print(f"  {metric}: {value}")

if __name__ == "__main__":
    main()
