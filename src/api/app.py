"""
Fraud Detection API for real-time inference.
This module provides a Flask API for serving the fraud detection models.
"""

import os
import json
import time
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import mlflow
import tensorflow as tf

# Configure TensorFlow to use GPU if available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth for all GPUs to prevent TensorFlow from allocating all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger = get_logger("api") if 'get_logger' in locals() else None

        # Print prominent GPU message
        gpu_message = f"🚀 GPU ACCELERATION ENABLED: Using {len(gpus)} GPU(s) for API inference"
        border = "="*70

        if logger:
            logger.info("\n" + border)
            logger.info(gpu_message)
            logger.info(border + "\n")
        else:
            print("\n" + border)
            print(gpu_message)
            print(border + "\n")

        # Test GPU performance
        import time
        test_message = "Testing GPU performance for API inference..."
        if logger:
            logger.info(test_message)
        else:
            print(test_message)

        x = tf.random.normal([1000, 1000])
        start = time.time()
        tf.matmul(x, x)
        perf_message = f"Matrix multiplication took: {time.time() - start:.6f} seconds"
        version_message = f"TensorFlow version: {tf.__version__}"

        if logger:
            logger.info(perf_message)
            logger.info(version_message)
        else:
            print(perf_message)
            print(version_message)

    except RuntimeError as e:
        error_message = f"Error configuring GPUs: {e}"
        if logger:
            logger.error(error_message)
        else:
            print(error_message)
else:
    # Print CPU fallback message
    cpu_message = "⚠️ NO GPU DETECTED: Using CPU for API inference (this will be slower)"
    border = "="*70

    logger = get_logger("api") if 'get_logger' in locals() else None
    if logger:
        logger.warning("\n" + border)
        logger.warning(cpu_message)
        logger.warning(border + "\n")
    else:
        print("\n" + border)
        print(cpu_message)
        print(border + "\n")
from prometheus_flask_exporter import PrometheusMetrics

# Import project modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logging_config import get_logger
from utils.config_manager import load_config

# Initialize Flask app
app = Flask(__name__)
metrics = PrometheusMetrics(app)

# Configure logging
logger = get_logger("api")

# Load configuration
config_path = os.environ.get("CONFIG_PATH", "config.production.yaml")
config = load_config(config_path)

# Load models
classification_model = None
autoencoder_model = None
feature_names = None
threshold = None

def load_models():
    """Load the trained models from MLflow."""
    global classification_model, autoencoder_model, feature_names, threshold

    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

        # Get the latest model versions
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(config["mlflow"]["experiment_name"])

        if experiment is None:
            logger.error(f"Experiment {config['mlflow']['experiment_name']} not found")
            return False

        # Get the latest run with models
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="attributes.status = 'FINISHED'",
            order_by=["attributes.start_time DESC"],
            max_results=1
        )

        if not runs:
            logger.error("No finished runs found")
            return False

        run_id = runs[0].info.run_id

        # Load models from the run
        artifacts_path = f"runs:/{run_id}"

        # Load classification model
        classification_model = mlflow.tensorflow.load_model(f"{artifacts_path}/classification_model")

        # Load autoencoder model
        autoencoder_model = mlflow.tensorflow.load_model(f"{artifacts_path}/autoencoder_model")

        # Load feature names
        feature_names_path = mlflow.artifacts.download_artifacts(f"{artifacts_path}/feature_names.json")
        with open(feature_names_path, "r") as f:
            feature_names = json.load(f)

        # Load threshold
        threshold_path = mlflow.artifacts.download_artifacts(f"{artifacts_path}/threshold.json")
        with open(threshold_path, "r") as f:
            threshold_data = json.load(f)
            threshold = threshold_data.get("threshold")

        logger.info("Models loaded successfully")
        return True

    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        return False

# Metrics
request_counter = metrics.counter(
    'fraud_detection_requests_total', 'Total number of fraud detection requests',
    labels={'status': lambda r: r.status_code}
)
prediction_histogram = metrics.histogram(
    'fraud_detection_prediction_time', 'Time spent processing prediction',
    labels={'model': lambda: 'fraud_detection'}
)

@app.before_first_request
def initialize():
    """Initialize the API by loading models."""
    if not load_models():
        logger.error("Failed to initialize models")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    if classification_model is None or autoencoder_model is None:
        return jsonify({"status": "error", "message": "Models not loaded"}), 503
    return jsonify({"status": "ok"}), 200

@app.route('/reload', methods=['POST'])
def reload_models():
    """Reload models endpoint."""
    success = load_models()
    if success:
        return jsonify({"status": "ok", "message": "Models reloaded successfully"}), 200
    else:
        return jsonify({"status": "error", "message": "Failed to reload models"}), 500

@app.route('/predict', methods=['POST'])
@request_counter
@prediction_histogram
def predict():
    """
    Predict endpoint for fraud detection.

    Expects a JSON payload with transaction data.
    Returns fraud prediction results.
    """
    start_time = time.time()

    # Check if models are loaded
    if classification_model is None or autoencoder_model is None:
        return jsonify({
            "status": "error",
            "message": "Models not loaded"
        }), 503

    # Get request data
    data = request.get_json()

    if not data:
        return jsonify({
            "status": "error",
            "message": "No data provided"
        }), 400

    try:
        # Convert to DataFrame
        if isinstance(data, list):
            # Multiple transactions
            df = pd.DataFrame(data)
        else:
            # Single transaction
            df = pd.DataFrame([data])

        # Check for required features
        missing_features = [f for f in feature_names if f not in df.columns]
        if missing_features:
            return jsonify({
                "status": "error",
                "message": f"Missing features: {', '.join(missing_features)}"
            }), 400

        # Ensure correct feature order
        df = df[feature_names]

        # Convert to numpy array
        X = df.values

        # Make predictions
        classification_pred = classification_model.predict(X)
        autoencoder_pred = autoencoder_model.predict(X)

        # Calculate anomaly scores
        mse = np.mean(np.square(X - autoencoder_pred), axis=1)

        # Prepare results
        results = []
        for i in range(len(X)):
            is_fraud_classification = bool(classification_pred[i][0] > 0.5)
            is_fraud_autoencoder = bool(mse[i] > threshold) if threshold else None

            # Combine results (if both models predict fraud, it's more likely to be fraud)
            is_fraud = is_fraud_classification or is_fraud_autoencoder

            results.append({
                "transaction_id": data[i].get("transaction_id", f"tx_{i}"),
                "fraud_probability": float(classification_pred[i][0]),
                "anomaly_score": float(mse[i]),
                "is_fraud_classification": is_fraud_classification,
                "is_fraud_autoencoder": is_fraud_autoencoder,
                "is_fraud": is_fraud
            })

        logger.info(f"Processed {len(results)} transactions in {time.time() - start_time:.2f}s")

        return jsonify({
            "status": "success",
            "predictions": results,
            "processing_time": time.time() - start_time
        }), 200

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Error processing request: {str(e)}"
        }), 500

if __name__ == '__main__':
    # Load models at startup
    load_models()

    # Run the app
    host = config["api"]["host"]
    port = config["api"]["port"]
    logger.info(f"Starting Fraud Detection API on {host}:{port}")
    app.run(host=host, port=port)
