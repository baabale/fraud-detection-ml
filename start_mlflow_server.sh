#!/bin/bash

# Create a new directory specifically for MLflow tracking server
mkdir -p mlflow_tracking

# Ensure proper permissions
chmod -R 755 mlflow_tracking

echo "Starting MLflow tracking server on http://localhost:5000"
echo "Artifacts will be stored in ./mlflow_tracking"
echo "Press Ctrl+C to stop the server"
echo "======================================================="

# Start the MLflow tracking server with the new directory
mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri ./mlflow_tracking \
    --default-artifact-root ./mlflow_tracking
