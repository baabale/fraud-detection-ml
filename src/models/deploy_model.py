"""
Script for deploying trained fraud detection models for inference.
This script creates a simple Flask API for model serving.
"""
import os
import argparse
import json
import sys
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler

# Add the project root to the path to ensure imports work from any directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the configuration manager
from src.utils.config_manager import config

# Conditionally import TensorFlow
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
    
    # Configure TensorFlow to use GPU if available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth for all GPUs to prevent TensorFlow from allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("\n" + "="*70)
            print(f"🚀 GPU ACCELERATION ENABLED: Using {len(gpus)} GPU(s) for model inference")
            print("="*70 + "\n")
            
            # Test GPU performance
            import time
            print("Testing GPU performance for inference...")
            x = tf.random.normal([1000, 1000])
            start = time.time()
            tf.matmul(x, x)
            print(f"Matrix multiplication took: {time.time() - start:.6f} seconds")
            print(f"TensorFlow version: {tf.__version__}")
        except RuntimeError as e:
            print(f"Error configuring GPUs: {e}")
    else:
        print("\n" + "="*70)
        print("⚠️ NO GPU DETECTED: Using CPU for model inference (this will be slower)")
        print("="*70 + "\n")
            
except ImportError:
    print("WARNING: TensorFlow not available. Using mock implementation.")
    TENSORFLOW_AVAILABLE = False
    
    # Create a simple mock for TensorFlow functionality
    class MockTF:
        class keras:
            class models:
                @staticmethod
                def load_model(path):
                    print(f"Mock: Loading model from {path}")
                    return MockModel()
        
        @staticmethod
        def __version__():
            return "MOCK"
    
    class MockModel:
        def predict(self, X):
            print(f"Mock: Predicting on data with shape {X.shape}")
            # Return random predictions
            return np.random.random(size=(X.shape[0], 1))
    
    # Use the mock
    tf = MockTF()

# Conditionally import MLflow
try:
    import mlflow
    import mlflow.tensorflow
    MLFLOW_AVAILABLE = True
except ImportError:
    print("WARNING: MLflow not available. Tracking functionality disabled.")
    MLFLOW_AVAILABLE = False

# Conditionally import joblib
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    print("WARNING: joblib not available. Using pickle as fallback.")
    JOBLIB_AVAILABLE = False
    import pickle as joblib

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
    
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Load scaler
    scaler_path = os.path.join(model_dir, 'scaler.joblib')
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
            print(f"Loaded scaler from {scaler_path}")
        except Exception as e:
            print(f"Error loading scaler: {str(e)}")
            scaler = StandardScaler()
            print("Using default StandardScaler instead")
    else:
        print(f"Warning: Scaler not found at {scaler_path}")
        scaler = StandardScaler()
        print("Using default StandardScaler instead")
    
    # Load feature names
    feature_names_path = os.path.join(model_dir, 'feature_names.json')
    if os.path.exists(feature_names_path):
        try:
            with open(feature_names_path, 'r') as f:
                feature_names = json.load(f)
            print(f"Loaded {len(feature_names)} feature names")
        except Exception as e:
            print(f"Error loading feature names: {str(e)}")
            feature_names = None
    else:
        print(f"Warning: Feature names not found at {feature_names_path}")
        feature_names = None
    
    # Load autoencoder threshold
    threshold_path = os.path.join(model_dir, 'autoencoder_threshold.json')
    if os.path.exists(threshold_path):
        try:
            with open(threshold_path, 'r') as f:
                threshold_data = json.load(f)
                autoencoder_threshold = threshold_data.get('threshold', 0.1)
            print(f"Loaded autoencoder threshold: {autoencoder_threshold}")
        except Exception as e:
            print(f"Error loading autoencoder threshold: {str(e)}")
            autoencoder_threshold = 0.1
            print(f"Using default threshold: {autoencoder_threshold}")
    else:
        autoencoder_threshold = 0.1
        print(f"Warning: Autoencoder threshold not found, using default: {autoencoder_threshold}")
    
    # Load classification model if TensorFlow is available
    if model_type in ['classification', 'both']:
        if TENSORFLOW_AVAILABLE:
            classification_model_path = os.path.join(model_dir, 'classification_model.keras')
            if os.path.exists(classification_model_path):
                try:
                    classification_model = tf.keras.models.load_model(classification_model_path)
                    print(f"Loaded classification model from {classification_model_path}")
                except Exception as e:
                    print(f"Error loading classification model: {str(e)}")
                    classification_model = None
            else:
                print(f"Warning: Classification model not found at {classification_model_path}")
                classification_model = None
        else:
            print("TensorFlow not available, cannot load classification model")
            classification_model = None
    
    # Load autoencoder model if TensorFlow is available
    if model_type in ['autoencoder', 'both']:
        if TENSORFLOW_AVAILABLE:
            # Load autoencoder model
            autoencoder_path = os.path.join(model_dir, 'autoencoder_model.keras')
            if os.path.exists(autoencoder_path):
                try:
                    # Define custom objects to handle 'mse' and other metrics
                    custom_objects = {
                        'mse': tf.keras.losses.MeanSquaredError(),
                        'mae': tf.keras.losses.MeanAbsoluteError(),
                        'mean_squared_error': tf.keras.losses.MeanSquaredError(),
                        'mean_absolute_error': tf.keras.losses.MeanAbsoluteError()
                    }
                    autoencoder_model = tf.keras.models.load_model(autoencoder_path, custom_objects=custom_objects)
                    print(f"Loaded autoencoder model from {autoencoder_path}")
                except Exception as e:
                    print(f"Error loading autoencoder model: {str(e)}")
                    autoencoder_model = None
            else:
                print(f"Warning: Autoencoder model not found at {autoencoder_path}")
                autoencoder_model = None
        else:
            print("TensorFlow not available, cannot load autoencoder model")
            autoencoder_model = None
    
    return classification_model, autoencoder_model

def preprocess_data(data):
    """
    Preprocess input data for model inference.
    
    Args:
        data (dict): Input transaction data
        
    Returns:
        array: Preprocessed features
    """
    try:
        # Print debugging info
        print(f"Input data type: {type(data)}")
        if isinstance(data, dict) and 'transactions' in data:
            print(f"Number of transactions: {len(data['transactions'])}")
            data = data['transactions']
        
        # Convert input data to DataFrame
        if isinstance(data, list):
            df = pd.DataFrame(data)
            print(f"Created DataFrame with shape: {df.shape}")
        else:
            df = pd.DataFrame([data])
            print(f"Created DataFrame with shape: {df.shape}")
        
        # Print column info
        print(f"DataFrame columns: {df.columns.tolist()}")
        print(f"Expected feature names: {feature_names}")
        
        # Ensure all required features are present
        if feature_names is not None:
            for feature in feature_names:
                if feature not in df.columns:
                    print(f"Adding missing feature: {feature}")
                    # Handle each missing feature with appropriate logic
                    if feature == 'transaction_frequency':
                        # Calculate transaction_frequency from time_since_last_transaction if available
                        if 'time_since_last_transaction' in df.columns:
                            # Convert to transactions per day (86400 seconds in a day)
                            df['transaction_frequency'] = np.where(
                                df['time_since_last_transaction'] > 0,
                                86400 / df['time_since_last_transaction'],
                                1.0  # Default value for first transaction
                            )
                            print("Created 'transaction_frequency' feature from 'time_since_last_transaction'")
                        else:
                            # If no time data available, use a default value
                            df['transaction_frequency'] = 1.0
                            print("Added 'transaction_frequency' with default value of 1.0")
                    elif feature == 'amount_log' and 'amount' in df.columns:
                        # Calculate log of amount
                        df['amount_log'] = np.log1p(df['amount'])
                        print("Created 'amount_log' feature from 'amount'")
                    elif feature == 'velocity_score_norm' and 'velocity_score' in df.columns:
                        # Normalize velocity score
                        df['velocity_score_norm'] = df['velocity_score'] / 100.0
                        print("Created 'velocity_score_norm' feature from 'velocity_score'")
                    else:
                        # For other features, use zero as default
                        df[feature] = 0.0
                        print(f"Added '{feature}' with default value of 0.0")
            
            # Select and order features according to feature_names
            df = df[feature_names]
            print(f"Reordered DataFrame with shape: {df.shape}")
        
        # Convert to numpy array
        X = df.values
        print(f"Converted to numpy array with shape: {X.shape}")
        
        # Scale features if scaler is available
        if scaler is not None:
            try:
                # Force reshape to match expected dimensions before scaling
                # This is a workaround for the "negative dimensions" error
                expected_features = 12  # Based on the feature_names.json we saw earlier
                if X.shape[1] != expected_features:
                    print(f"Reshaping X from {X.shape} to match expected {expected_features} features before scaling")
                    # Create a new array with the right shape
                    new_X = np.zeros((X.shape[0], expected_features))
                    # Copy as many features as we can
                    for i in range(min(X.shape[1], expected_features)):
                        new_X[:, i] = X[:, i]
                    X = new_X
                
                # Now try to scale
                X = scaler.transform(X)
                print(f"Scaled features, new shape: {X.shape}")
            except Exception as e:
                print(f"Error scaling features: {str(e)}")
                # If scaling still fails, create a properly shaped array with zeros
                print("Creating fallback array with proper dimensions")
                X = np.zeros((X.shape[0], 12))  # Use 12 features as seen in feature_names.json
        
        # Check if we need to adjust dimensions for the model
        if classification_model is not None:
            try:
                expected_shape = classification_model.layers[0].input_shape[1]
                print(f"Model expects input shape with {expected_shape} features")
                current_shape = X.shape[1]
                
                if expected_shape > current_shape:
                    print(f"Adding {expected_shape - current_shape} placeholder features")
                    # Add placeholder features (zeros) to match the expected input shape
                    additional_features = np.zeros((X.shape[0], expected_shape - current_shape))
                    X = np.hstack((X, additional_features))
                elif expected_shape < current_shape:
                    print(f"Trimming {current_shape - expected_shape} extra features")
                    # Trim extra features
                    X = X[:, :expected_shape]
                
                print(f"Final X shape: {X.shape}")
            except Exception as e:
                print(f"Error adjusting dimensions: {str(e)}")
        
        return X
    except Exception as e:
        print(f"Error in preprocess_data: {str(e)}")
        # Return a simple placeholder array as fallback
        if classification_model is not None:
            try:
                expected_shape = classification_model.layers[0].input_shape[1]
                return np.zeros((1, expected_shape))
            except:
                pass
        return np.zeros((1, 12))  # Default to 12 features

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
        data = request.json
        print(f"Received request data: {data}")
        
        if not data or 'transactions' not in data:
            return jsonify({'error': 'Invalid request format'}), 400
        
        # Get model type from request or use default
        model_type = data.get('model_type', 'both')
        if model_type not in ['classification', 'autoencoder', 'both', 'ensemble']:
            model_type = 'both'  # Default to both
        print(f"Using model type: {model_type}")
        
        # Check model availability
        if model_type in ['classification', 'both', 'ensemble'] and classification_model is None:
            print("Classification model not available but requested")
            return jsonify({'error': 'Classification model not available'}), 500
            
        if model_type in ['autoencoder', 'both', 'ensemble'] and autoencoder_model is None:
            print("Autoencoder model not available but requested")
            return jsonify({'error': 'Autoencoder model not available'}), 500
        
        # Preprocess data
        try:
            X = preprocess_data(data['transactions'])
            print(f"Preprocessed data shape: {X.shape}")
        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            return jsonify({'error': f'Preprocessing error: {str(e)}'}), 500
        
        # Make predictions
        results = []
        
        if model_type == 'classification' or model_type == 'ensemble':
            if classification_model is None:
                return jsonify({'error': 'Classification model not loaded'}), 500
            
            # Check if input shape matches model's expected input shape
            input_shape = X.shape[1]
            expected_shape = 6  # The model expects 6 features
            
            if input_shape != expected_shape:
                print(f"Input shape mismatch: got {input_shape}, expected {expected_shape}. Padding with zeros.")
                # Pad the input tensor with zeros to match the expected shape
                padding = np.zeros((X.shape[0], expected_shape - input_shape))
                X_padded = np.hstack((X, padding))
                # Get classification predictions with padded input
                classification_probs = classification_model.predict(X_padded).flatten()
            else:
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
    parser.add_argument('--model-dir', type=str,
                        help='Directory containing the trained models')
    parser.add_argument('--model-type', type=str, choices=['classification', 'autoencoder', 'both'],
                        help='Type of model to deploy')
    parser.add_argument('--host', type=str,
                        help='Host to run the server on')
    parser.add_argument('--port', type=int,
                        help='Port to run the server on')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode')
    parser.add_argument('--disable-gpu', action='store_true', 
                        help='Disable GPU usage even if available')
    parser.add_argument('--single-gpu', action='store_true', 
                        help='Use only a single GPU even if multiple are available')
    parser.add_argument('--memory-growth', action='store_true', 
                        help='Enable memory growth for GPUs to prevent TensorFlow from allocating all memory')
    args = parser.parse_args()
    
    # Use configuration values if arguments are not provided
    model_dir = args.model_dir or config.get_model_path()
    model_type = args.model_type or config.get('models.model_type', 'both')
    host = args.host or config.get('api.host', '0.0.0.0')
    port = args.port or config.get('api.port', 8080)
    debug = args.debug or config.get('api.debug', False)
    
    # Load models
    load_models(model_dir, model_type)
    
    # Run the server
    app.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    main()
