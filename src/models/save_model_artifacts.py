"""
Script to save model artifacts for deployment.
This script saves the trained models along with necessary metadata.
"""
import os
import argparse
import json
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

def save_model_artifacts(model_path, data_path, output_dir, threshold_percentile=95):
    """
    Save model artifacts for deployment.
    
    Args:
        model_path (str): Path to the trained model
        data_path (str): Path to the processed data
        output_dir (str): Directory to save artifacts
        threshold_percentile (int): Percentile for autoencoder threshold
    """
    print(f"Loading model from {model_path}")
    print(f"Loading data from {data_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    elif data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
    
    # Separate features and target
    if 'is_fraud' in df.columns:
        y = df['is_fraud'].values
        X = df.drop(columns=['is_fraud'])
    else:
        raise ValueError("Target column 'is_fraud' not found in data")
    
    # Save feature names
    feature_names = X.columns.tolist()
    with open(os.path.join(output_dir, 'feature_names.json'), 'w') as f:
        json.dump(feature_names, f)
    print(f"Saved {len(feature_names)} feature names")
    
    # Fit and save scaler
    scaler = StandardScaler()
    scaler.fit(X)
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.joblib'))
    print("Saved feature scaler")
    
    # Load and save model
    model_filename = os.path.basename(model_path)
    model = tf.keras.models.load_model(model_path)
    model.save(os.path.join(output_dir, model_filename))
    print(f"Saved model to {os.path.join(output_dir, model_filename)}")
    
    # If it's an autoencoder model, calculate and save threshold
    if 'autoencoder' in model_path:
        # Scale the data
        X_scaled = scaler.transform(X)
        
        # Get reconstructions
        X_pred = model.predict(X_scaled)
        
        # Compute mean squared error for each sample
        mse = np.mean(np.square(X_scaled - X_pred), axis=1)
        
        # Calculate threshold based on non-fraud transactions
        non_fraud_indices = (y == 0)
        non_fraud_scores = mse[non_fraud_indices]
        threshold = np.percentile(non_fraud_scores, threshold_percentile)
        
        # Save threshold
        with open(os.path.join(output_dir, 'autoencoder_threshold.json'), 'w') as f:
            json.dump({'threshold': float(threshold), 'percentile': threshold_percentile}, f)
        print(f"Saved autoencoder threshold: {threshold:.4f} (percentile: {threshold_percentile})")
    
    print(f"All artifacts saved to {output_dir}")

def main():
    """
    Main function to save model artifacts.
    """
    parser = argparse.ArgumentParser(description='Save model artifacts for deployment')
    parser.add_argument('--classification-model', type=str,
                        default='../../results/models/classification_model.h5',
                        help='Path to the classification model')
    parser.add_argument('--autoencoder-model', type=str,
                        default='../../results/models/autoencoder_model.h5',
                        help='Path to the autoencoder model')
    parser.add_argument('--data-path', type=str,
                        default='../../data/processed/transactions.parquet',
                        help='Path to the processed data')
    parser.add_argument('--output-dir', type=str,
                        default='../../results/deployment',
                        help='Directory to save artifacts')
    parser.add_argument('--threshold-percentile', type=int, default=95,
                        help='Percentile for autoencoder threshold')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save classification model artifacts
    if os.path.exists(args.classification_model):
        save_model_artifacts(
            args.classification_model,
            args.data_path,
            args.output_dir,
            args.threshold_percentile
        )
    else:
        print(f"Classification model not found: {args.classification_model}")
    
    # Save autoencoder model artifacts
    if os.path.exists(args.autoencoder_model):
        save_model_artifacts(
            args.autoencoder_model,
            args.data_path,
            args.output_dir,
            args.threshold_percentile
        )
    else:
        print(f"Autoencoder model not found: {args.autoencoder_model}")

if __name__ == "__main__":
    main()
