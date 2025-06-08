#!/bin/bash
set -e

# Script to clean up all model artifacts and metrics for fresh training
echo "Cleaning up model artifacts and metrics for fresh training..."

# Clean up results directory (keeping directory structure)
echo "Cleaning results/models directory..."
rm -rf results/models/*.keras results/models/*.json results/models/*.joblib
echo "Cleaning results/metrics directory..."
rm -rf results/metrics/*.json

# Clean up any figures
echo "Cleaning results/figures directory..."
rm -rf results/figures/*.png

# If using MLflow, clean up local mlruns directory if it exists
if [ -d "mlruns" ]; then
    echo "Cleaning local mlruns directory..."
    rm -rf mlruns/*
fi

echo "Cleanup complete! Ready for fresh training."
