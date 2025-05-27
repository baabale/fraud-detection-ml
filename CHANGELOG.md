# Changelog

All notable changes to the Anomaly-Based Fraud Detection project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- API documentation with detailed endpoint descriptions
- Enhanced model deployment with input shape handling
- Fixed port conflict issues in API deployment
- Added test script for API validation
- Added GPU-related command-line arguments to deployment scripts
- Added improved error handling and debugging in API server
- Added GPU-related command-line arguments to streaming fraud detection script

### Fixed
- Resolved NaN handling in model training and evaluation
- Fixed model file path issues in deployment scripts
- Addressed input feature mismatch between training and deployment
- Fixed GPU parameter handling in model deployment scripts
- Fixed non-numeric column handling in model artifact saving
- Added custom objects support for loading models with custom metrics
- Fixed GPU parameter handling in streaming fraud detection script

## [1.0.0] - 2025-05-25

### Added
- Initial project setup with complete directory structure
- Data processing pipeline using PySpark
- Classification model for supervised fraud detection
- Autoencoder model for unsupervised anomaly detection
- Model training pipeline with MLflow integration
- Model evaluation script with comprehensive metrics
- Interactive user interface via command-line menu
- Deployment capabilities with Flask REST API
- Real-time fraud detection using Spark Structured Streaming
- Model monitoring and drift detection system
- Synthetic data generation utility
- Jupyter notebooks for exploratory data analysis
- Configuration system with YAML files
- MIT License
- Comprehensive README with badges and documentation
- Convenient run script for easy setup and execution
- GitHub repository setup

### Technical Details
- PySpark data processing with feature engineering
- TensorFlow deep learning models (classification and autoencoder)
- MLflow experiment tracking and model management
- Comprehensive evaluation metrics and visualizations
- Flask API for model deployment
- Spark Structured Streaming for real-time processing
- Data drift detection and model performance monitoring
- Interactive command-line interface
- Virtual environment management

## [0.1.0] - 2025-05-25

### Added
- Initial project concept and design
- Project structure planning
- Technology selection
- Initial documentation in PROJECT.md

[Unreleased]: https://github.com/baabale/fraud-detection-ml/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/baabale/fraud-detection-ml/releases/tag/v1.0.0
[0.1.0]: https://github.com/baabale/fraud-detection-ml/releases/tag/v0.1.0
