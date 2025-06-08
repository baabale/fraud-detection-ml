#!/usr/bin/env python3
"""
Test script for evaluating the enhanced fraud detection model evaluation functions.
This script creates synthetic data and models to test the evaluation functions.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import json
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Define compute_anomaly_scores function to avoid dependency issues
def compute_anomaly_scores(model, X):
    """Compute anomaly scores for autoencoder model"""
    # Get reconstructed data
    X_reconstructed = model.predict(X)
    
    # Calculate reconstruction error (MSE) for each sample
    mse = np.mean(np.square(X - X_reconstructed), axis=1)
    
    return mse

# Import our evaluation functions
from src.models.evaluate_model import (
    evaluate_classification_model,
    evaluate_autoencoder_model,
    analyze_feature_importance,
    generate_classification_visualizations
)

# Monkey patch the compute_anomaly_scores function
import sys
sys.modules['src.models.fraud_model'] = type('fraud_model', (), {'compute_anomaly_scores': compute_anomaly_scores})


# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Create output directories
os.makedirs('results/test_evaluation/figures', exist_ok=True)
os.makedirs('results/test_evaluation/metrics', exist_ok=True)

def create_synthetic_data(n_samples=1000, n_features=20, fraud_ratio=0.1):
    """Create synthetic fraud detection data"""
    print("Creating synthetic fraud detection data...")
    
    # Generate feature data
    X = np.random.randn(n_samples, n_features)
    
    # Create some patterns for fraudulent transactions
    # Fraud transactions have higher values in certain features
    fraud_pattern = np.zeros(n_features)
    fraud_pattern[0:5] = 2.0  # First 5 features are important for fraud
    
    # Generate labels with imbalance
    n_fraud = int(n_samples * fraud_ratio)
    n_normal = n_samples - n_fraud
    
    y = np.zeros(n_samples)
    fraud_indices = np.random.choice(n_samples, n_fraud, replace=False)
    y[fraud_indices] = 1
    
    # Apply fraud pattern to fraudulent transactions
    for idx in fraud_indices:
        X[idx] += fraud_pattern + np.random.randn(n_features) * 0.5
    
    # Create feature names
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
    
    print(f"Created dataset with {n_samples} samples, {n_fraud} fraudulent ({fraud_ratio*100:.1f}%)")
    print(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test, feature_names

def create_classification_model(X_train, y_train):
    """Create and train a simple classification model"""
    print("Creating and training classification model...")
    
    # Create a simple Random Forest model
    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    return model

def create_autoencoder_model(X_train, input_dim):
    """Create and train a simple autoencoder model"""
    print("Creating and training autoencoder model...")
    
    # Create a simple autoencoder
    input_layer = tf.keras.layers.Input(shape=(input_dim,))
    encoded = tf.keras.layers.Dense(10, activation='relu')(input_layer)
    decoded = tf.keras.layers.Dense(input_dim, activation='linear')(encoded)
    
    autoencoder = tf.keras.models.Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    # Train the model
    autoencoder.fit(
        X_train, X_train,
        epochs=10,
        batch_size=32,
        shuffle=True,
        verbose=1
    )
    
    return autoencoder

def test_classification_evaluation():
    """Test the classification model evaluation function"""
    print("\n=== Testing Classification Model Evaluation ===")
    
    # Create synthetic data
    X_train, X_test, y_train, y_test, feature_names = create_synthetic_data()
    
    # Train a classification model
    model = create_classification_model(X_train, y_train)
    
    # Evaluate the model
    metrics = evaluate_classification_model(
        model, X_test, y_test,
        feature_names=feature_names,
        threshold=0.5,
        avg_transaction_amount=500,
        cost_fp_ratio=0.1,
        cost_fn_ratio=1.0,
        output_dir='results/test_evaluation/figures'
    )
    
    # Manually call generate_classification_visualizations with correct parameters
    # Get predictions for visualization
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Skip manual visualization generation since it's already done in evaluate_classification_model
    
    # Save metrics to file
    with open('results/test_evaluation/metrics/classification_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print("Classification evaluation complete. Metrics saved to results/test_evaluation/metrics/classification_metrics.json")
    
    return metrics

def test_autoencoder_evaluation():
    """Test the autoencoder model evaluation function"""
    print("\n=== Testing Autoencoder Model Evaluation ===")
    
    # Create synthetic data
    X_train, X_test, y_train, y_test, feature_names = create_synthetic_data()
    
    # Train an autoencoder model
    model = create_autoencoder_model(X_train, X_train.shape[1])
    
    # Evaluate the model
    metrics, anomaly_scores = evaluate_autoencoder_model(
        model, X_test, y_test,
        feature_names=feature_names,
        threshold=None,
        percentile=90,
        optimize_threshold=True,
        output_dir='results/test_evaluation/figures'
    )
    
    # Save metrics to file
    with open('results/test_evaluation/metrics/autoencoder_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print("Autoencoder evaluation complete. Metrics saved to results/test_evaluation/metrics/autoencoder_metrics.json")
    
    return metrics, anomaly_scores

def main():
    """Main function to run the tests"""
    print("Starting evaluation function tests...")
    
    # Test classification model evaluation
    classification_metrics = test_classification_evaluation()
    
    # Test autoencoder model evaluation
    autoencoder_metrics, _ = test_autoencoder_evaluation()
    
    print("\n=== Test Summary ===")
    print("Classification Model Performance:")
    print(f"  Accuracy: {classification_metrics['accuracy']:.4f}")
    print(f"  Precision: {classification_metrics['precision']:.4f}")
    print(f"  Recall: {classification_metrics['recall']:.4f}")
    print(f"  F1 Score: {classification_metrics['f1_score']:.4f}")
    
    print("\nAutoencoder Model Performance:")
    print(f"  Accuracy: {autoencoder_metrics['accuracy']:.4f}")
    print(f"  Precision: {autoencoder_metrics['precision']:.4f}")
    print(f"  Recall: {autoencoder_metrics['recall']:.4f}")
    print(f"  F1 Score: {autoencoder_metrics['f1_score']:.4f}")
    
    print("\nEvaluation visualizations saved to results/test_evaluation/figures")
    print("Evaluation metrics saved to results/test_evaluation/metrics")

if __name__ == "__main__":
    main()
