"""
Script for deploying trained fraud detection models for inference.
This script creates a simple Flask API for model serving.
"""
import os
import argparse
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, jsonify
import mlflow
import mlflow.tensorflow
from sklearn.preprocessing import StandardScaler
import joblib

# Create Flask app
app = Flask(__name__)

# Global variables for loaded models and preprocessors
classification_model = None
autoencoder_model = None
scaler = None
feature_names = None
autoencoder_threshold = None

def load_models(model_dir, model_type='both'):
    """
    Load trained models from disk.
    
    Args:
        model_dir (str): Directory containing the models
        model_type (str): Type of model to load ('classification', 'autoencoder', or 'both')
        
    Returns:
        tuple: Loaded models
    """
    global classification_model, autoencoder_model, scaler, feature_names, autoencoder_threshold
    
    # Load scaler
    scaler_path = os.path.join(model_dir, 'scaler.joblib')
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print(f"Loaded scaler from {scaler_path}")
    else:
        print(f"Warning: Scaler not found at {scaler_path}")
    
    # Load feature names
    feature_names_path = os.path.join(model_dir, 'feature_names.json')
    if os.path.exists(feature_names_path):
        with open(feature_names_path, 'r') as f:
            feature_names = json.load(f)
        print(f"Loaded {len(feature_names)} feature names")
    else:
        print(f"Warning: Feature names not found at {feature_names_path}")
    
    # Load autoencoder threshold
    threshold_path = os.path.join(model_dir, 'autoencoder_threshold.json')
    if os.path.exists(threshold_path):
        with open(threshold_path, 'r') as f:
            threshold_data = json.load(f)
            autoencoder_threshold = threshold_data.get('threshold', 0.1)
        print(f"Loaded autoencoder threshold: {autoencoder_threshold}")
    else:
        autoencoder_threshold = 0.1
        print(f"Warning: Autoencoder threshold not found, using default: {autoencoder_threshold}")
    
    # Load classification model
    if model_type in ['classification', 'both']:
        classification_model_path = os.path.join(model_dir, 'classification_model.h5')
        if os.path.exists(classification_model_path):
            classification_model = tf.keras.models.load_model(classification_model_path)
            print(f"Loaded classification model from {classification_model_path}")
        else:
            print(f"Warning: Classification model not found at {classification_model_path}")
    
    # Load autoencoder model
    if model_type in ['autoencoder', 'both']:
        autoencoder_model_path = os.path.join(model_dir, 'autoencoder_model.h5')
        if os.path.exists(autoencoder_model_path):
            autoencoder_model = tf.keras.models.load_model(autoencoder_model_path)
            print(f"Loaded autoencoder model from {autoencoder_model_path}")
        else:
            print(f"Warning: Autoencoder model not found at {autoencoder_model_path}")
    
    return classification_model, autoencoder_model

def preprocess_data(data):
    """
    Preprocess input data for model inference.
    
    Args:
        data (dict): Input transaction data
        
    Returns:
        array: Preprocessed features
    """
    # Convert input data to DataFrame
    if isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        df = pd.DataFrame([data])
    
    # Ensure all required features are present
    if feature_names is not None:
        for feature in feature_names:
            if feature not in df.columns:
                df[feature] = 0  # Default value for missing features
        
        # Select and order features according to feature_names
        df = df[feature_names]
    
    # Convert to numpy array
    X = df.values
    
    # Scale features if scaler is available
    if scaler is not None:
        X = scaler.transform(X)
    
    return X

def compute_anomaly_scores(model, X):
    """
    Compute anomaly scores using the autoencoder reconstruction error.
    
    Args:
        model: Trained autoencoder model
        X: Input data
        
    Returns:
        array: Anomaly scores
    """
    # Get reconstructions
    X_pred = model.predict(X)
    
    # Compute mean squared error for each sample
    mse = np.mean(np.square(X - X_pred), axis=1)
    
    return mse

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint for fraud prediction.
    
    Request format:
    {
        "transactions": [
            {
                "feature1": value1,
                "feature2": value2,
                ...
            },
            ...
        ],
        "model_type": "classification" or "autoencoder" or "ensemble"
    }
    
    Response format:
    {
        "predictions": [
            {
                "fraud_probability": 0.xx,
                "is_fraud": true/false,
                "anomaly_score": 0.xx (if autoencoder or ensemble)
            },
            ...
        ]
    }
    """
    try:
        # Get request data
        data = request.get_json()
        
        if not data or 'transactions' not in data:
            return jsonify({'error': 'No transaction data provided'}), 400
        
        # Get model type (default to ensemble if both models are available)
        model_type = data.get('model_type', 'ensemble')
        if model_type == 'ensemble' and (classification_model is None or autoencoder_model is None):
            if classification_model is not None:
                model_type = 'classification'
            elif autoencoder_model is not None:
                model_type = 'autoencoder'
            else:
                return jsonify({'error': 'No models available'}), 500
        
        # Preprocess data
        transactions = data['transactions']
        X = preprocess_data(transactions)
        
        # Make predictions
        results = []
        
        if model_type == 'classification' or model_type == 'ensemble':
            if classification_model is None:
                return jsonify({'error': 'Classification model not loaded'}), 500
            
            # Get classification predictions
            classification_probs = classification_model.predict(X).flatten()
            classification_preds = (classification_probs >= 0.5).astype(bool)
        
        if model_type == 'autoencoder' or model_type == 'ensemble':
            if autoencoder_model is None:
                return jsonify({'error': 'Autoencoder model not loaded'}), 500
            
            # Get anomaly scores
            anomaly_scores = compute_anomaly_scores(autoencoder_model, X)
            autoencoder_preds = (anomaly_scores >= autoencoder_threshold).astype(bool)
        
        # Combine results based on model type
        for i in range(len(X)):
            result = {}
            
            if model_type == 'classification':
                result = {
                    'fraud_probability': float(classification_probs[i]),
                    'is_fraud': bool(classification_preds[i])
                }
            elif model_type == 'autoencoder':
                result = {
                    'anomaly_score': float(anomaly_scores[i]),
                    'is_fraud': bool(autoencoder_preds[i])
                }
            elif model_type == 'ensemble':
                # Combine predictions (if either model predicts fraud, consider it fraud)
                is_fraud = bool(classification_preds[i] or autoencoder_preds[i])
                result = {
                    'fraud_probability': float(classification_probs[i]),
                    'anomaly_score': float(anomaly_scores[i]),
                    'is_fraud': is_fraud
                }
            
            results.append(result)
        
        return jsonify({'predictions': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint.
    """
    health_status = {
        'status': 'ok',
        'models': {
            'classification': classification_model is not None,
            'autoencoder': autoencoder_model is not None
        }
    }
    return jsonify(health_status)

def main():
    """
    Main function to run the model deployment server.
    """
    parser = argparse.ArgumentParser(description='Deploy fraud detection models as an API')
    parser.add_argument('--model-dir', type=str, default='../../results/models',
                        help='Directory containing the trained models')
    parser.add_argument('--model-type', type=str, choices=['classification', 'autoencoder', 'both'],
                        default='both', help='Type of model to deploy')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to run the server on')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port to run the server on')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode')
    args = parser.parse_args()
    
    # Load models
    load_models(args.model_dir, args.model_type)
    
    # Run the server
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
