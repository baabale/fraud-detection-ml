#!/bin/bash

# Script to run model training with or without MLflow tracking
# Author: Cascade AI

# Default values
USE_MLFLOW=false
MODEL_TYPE="classification"
START_MLFLOW=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --with-mlflow)
      USE_MLFLOW=true
      shift
      ;;
    --start-mlflow)
      START_MLFLOW=true
      USE_MLFLOW=true
      shift
      ;;
    --model-type)
      MODEL_TYPE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Create necessary directories
mkdir -p models
mkdir -p logs

# If requested, start MLflow server in the background
if [ "$START_MLFLOW" = true ]; then
  echo "Starting MLflow tracking server..."
  # Check if MLflow server is already running
  if nc -z localhost 5000 2>/dev/null; then
    echo "MLflow server is already running on port 5000"
  else
    # Start MLflow server in the background
    ./start_mlflow_server.sh > logs/mlflow_server.log 2>&1 &
    MLFLOW_PID=$!
    echo "MLflow server started with PID: $MLFLOW_PID"
    echo "Waiting for server to initialize..."
    sleep 5
  fi
fi

# Set up MLflow arguments
MLFLOW_ARGS=""
if [ "$USE_MLFLOW" = true ]; then
  MLFLOW_ARGS="--use-mlflow"
fi

# Run the training script
echo "Starting model training for $MODEL_TYPE model..."
python src/models/train_model.py \
  --data-path data/processed/transactions.parquet \
  --model-dir models \
  --model-type $MODEL_TYPE \
  $MLFLOW_ARGS

# Training complete
echo "Training complete!"
