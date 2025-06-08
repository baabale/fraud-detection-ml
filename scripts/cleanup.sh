#!/bin/bash
# Cleanup script for the fraud detection project
# This script removes cache files, system files, and other temporary files
# while preserving useful test scripts

echo "Starting cleanup process..."

# Remove Python cache files
echo "Removing Python cache files..."
find . -name "*.pyc" -type f -delete
find . -name "__pycache__" -type d -exec rm -rf {} +

# Remove macOS system files
echo "Removing macOS system files..."
find . -name ".DS_Store" -type f -delete

# Remove log files (optional - uncomment if you want to remove logs)
# echo "Removing log files..."
# find ./logs -name "*.log" -type f -delete

# Remove any temporary files
echo "Removing temporary files..."
find . -name "*.tmp" -type f -delete
find . -name "*.bak" -type f -delete

echo "Cleanup completed successfully!"
