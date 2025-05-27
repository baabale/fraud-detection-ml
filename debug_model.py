#!/usr/bin/env python
"""
Debug script to test model loading and prediction directly.
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Configure TensorFlow
try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    
    # Configure GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Found {len(gpus)} GPUs")
    else:
        print("No GPUs found, using CPU")
except ImportError:
    print("TensorFlow not available")

# Paths
model_dir = "results/deployment"
classification_model_path = os.path.join(model_dir, "classification_model.h5")
autoencoder_model_path = os.path.join(model_dir, "autoencoder_model.h5")
scaler_path = os.path.join(model_dir, "scaler.joblib")
feature_names_path = os.path.join(model_dir, "feature_names.json")

# Load feature names
try:
    with open(feature_names_path, 'r') as f:
        feature_names = json.load(f)
    print(f"Loaded feature names: {feature_names}")
except Exception as e:
    print(f"Error loading feature names: {e}")
    feature_names = []

# Load scaler
try:
    import joblib
    scaler = joblib.load(scaler_path)
    print("Loaded scaler successfully")
except Exception as e:
    print(f"Error loading scaler: {e}")
    scaler = None

# Load models
try:
    # Define custom objects for model loading
    custom_objects = {
        'mse': tf.keras.losses.MeanSquaredError(),
        'mae': tf.keras.losses.MeanAbsoluteError(),
        'mean_squared_error': tf.keras.losses.MeanSquaredError(),
        'mean_absolute_error': tf.keras.losses.MeanAbsoluteError()
    }
    
    # Load classification model
    if os.path.exists(classification_model_path):
        classification_model = tf.keras.models.load_model(classification_model_path, custom_objects=custom_objects)
        print(f"Loaded classification model from {classification_model_path}")
        
        # Print model summary
        classification_model.summary()
        
        # Get input shape
        input_shape = classification_model.layers[0].input_shape
        print(f"Classification model input shape: {input_shape}")
    else:
        print(f"Classification model not found at {classification_model_path}")
        classification_model = None
    
    # Load autoencoder model
    if os.path.exists(autoencoder_model_path):
        autoencoder_model = tf.keras.models.load_model(autoencoder_model_path, custom_objects=custom_objects)
        print(f"Loaded autoencoder model from {autoencoder_model_path}")
        
        # Print model summary
        autoencoder_model.summary()
        
        # Get input shape
        input_shape = autoencoder_model.layers[0].input_shape
        print(f"Autoencoder model input shape: {input_shape}")
    else:
        print(f"Autoencoder model not found at {autoencoder_model_path}")
        autoencoder_model = None
except Exception as e:
    print(f"Error loading models: {e}")
    classification_model = None
    autoencoder_model = None

# Create test data
test_data = {
    "amount": 500.0,
    "time_since_last_transaction": 3600,
    "spending_deviation_score": 0.2,
    "velocity_score": 0.3,
    "geo_anomaly_score": 0.1,
    "amount_log": 6.2,
    "hour_of_day": 14,
    "day_of_week": 3,
    "month": 5,
    "year": 2025,
    "fraud_label": 0,
    "velocity_score_norm": 0.3
}

# Convert to DataFrame
df = pd.DataFrame([test_data])
print(f"Created DataFrame with shape: {df.shape}")
print(f"DataFrame columns: {df.columns.tolist()}")

# Ensure all required features are present and in the right order
if feature_names:
    for feature in feature_names:
        if feature not in df.columns:
            print(f"Adding missing feature: {feature}")
            df[feature] = 0
    
    # Select and order features according to feature_names
    df = df[feature_names]
    print(f"Reordered DataFrame with shape: {df.shape}")

# Convert to numpy array
X = df.values
print(f"Converted to numpy array with shape: {X.shape}")

# Scale features if scaler is available
if scaler is not None:
    try:
        X_scaled = scaler.transform(X)
        print(f"Scaled features, new shape: {X_scaled.shape}")
    except Exception as e:
        print(f"Error scaling features: {e}")
        X_scaled = X
else:
    X_scaled = X

# Make predictions with classification model
if classification_model is not None:
    try:
        # Get expected input shape
        expected_shape = classification_model.layers[0].input_shape[1]
        print(f"Model expects input shape with {expected_shape} features")
        
        # Adjust dimensions if needed
        if X_scaled.shape[1] != expected_shape:
            print(f"Reshaping X from {X_scaled.shape} to match expected shape with {expected_shape} features")
            # Create a new array with the right shape
            new_X = np.zeros((X_scaled.shape[0], expected_shape))
            # Copy as many features as we can
            for i in range(min(X_scaled.shape[1], expected_shape)):
                new_X[:, i] = X_scaled[:, i]
            X_scaled = new_X
        
        # Make prediction
        pred = classification_model.predict(X_scaled)
        print(f"Classification prediction: {pred}")
        print(f"Is fraud: {bool(pred[0][0] >= 0.5)}")
    except Exception as e:
        print(f"Error making classification prediction: {e}")

# Make predictions with autoencoder model
if autoencoder_model is not None:
    try:
        # Get expected input shape
        expected_shape = autoencoder_model.layers[0].input_shape[1]
        print(f"Autoencoder expects input shape with {expected_shape} features")
        
        # Adjust dimensions if needed
        if X_scaled.shape[1] != expected_shape:
            print(f"Reshaping X from {X_scaled.shape} to match expected shape with {expected_shape} features")
            # Create a new array with the right shape
            new_X = np.zeros((X_scaled.shape[0], expected_shape))
            # Copy as many features as we can
            for i in range(min(X_scaled.shape[1], expected_shape)):
                new_X[:, i] = X_scaled[:, i]
            X_scaled = new_X
        
        # Make prediction
        reconstructed = autoencoder_model.predict(X_scaled)
        mse = np.mean(np.square(X_scaled - reconstructed), axis=1)
        print(f"Autoencoder reconstruction error: {mse}")
        
        # Load threshold
        threshold_path = os.path.join(model_dir, "autoencoder_threshold.json")
        if os.path.exists(threshold_path):
            with open(threshold_path, 'r') as f:
                threshold_data = json.load(f)
                threshold = threshold_data.get('threshold', 0.1)
            print(f"Loaded autoencoder threshold: {threshold}")
            print(f"Is anomaly: {bool(mse[0] >= threshold)}")
        else:
            print("Autoencoder threshold not found")
    except Exception as e:
        print(f"Error making autoencoder prediction: {e}")
