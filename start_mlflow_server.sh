#!/bin/bash

# Create the mlruns directory if it doesn't exist
mkdir -p mlruns

echo "Starting MLflow tracking server on http://localhost:5000"
echo "Artifacts will be stored in ./mlruns"
echo "Press Ctrl+C to stop the server"
echo "======================================================="

# Start the MLflow tracking server
mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri ./mlruns \
    --default-artifact-root ./mlruns
