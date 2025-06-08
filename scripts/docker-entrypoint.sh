#!/bin/bash
set -e

# Create necessary directories
mkdir -p /app/logs /app/data/raw /app/data/processed /app/results/models /app/results/metrics

# Check if the command is specified
if [ "$1" = "api" ]; then
    echo "Starting Fraud Detection API..."
    cd /app/src/api
    exec gunicorn --bind 0.0.0.0:8000 --workers 4 --timeout 120 wsgi:app
elif [ "$1" = "train" ]; then
    echo "Starting model training pipeline..."
    exec python /app/src/core/pipeline.py
elif [ "$1" = "stream" ]; then
    echo "Starting streaming fraud detection..."
    exec python /app/src/spark_jobs/streaming/streaming_fraud_detection.py
elif [ "$1" = "monitor" ]; then
    echo "Starting model monitoring..."
    exec python /app/src/models/monitoring/model_monitoring.py
else
    # Default to executing the passed command
    exec "$@"
fi
