# Changelog

All notable changes to the Anomaly-Based Fraud Detection project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Implemented cluster-based SMOTE sampling technique for better preservation of minority class distribution patterns
- Added cost-sensitive evaluation metrics to translate model performance into business value
- Added financial impact visualization tools to find cost-optimal classification thresholds
- Added cyclical time encoding features (hour, day, month) to capture time patterns in transactions
- Implemented transaction sequence features to track patterns in transaction behavior
- Enhanced model architecture with deeper neural networks for better feature learning
- Implemented L2 regularization to prevent overfitting in classification models
- Optimized sampling techniques with Borderline SMOTE for improved handling of class imbalance
- Added asymmetric focal loss function with tuned parameters for better fraud detection
- Improved autoencoder model with larger encoding dimension and regularization
- Added command-line support for L2 regularization parameter

## [2.0.0] - 2025-06-01

### Model Format Update

#### Overview
This update addresses the model format deprecation warnings by transitioning from the legacy HDF5 (`.h5`) format to the native Keras (`.keras`) format. The changes ensure compatibility with TensorFlow/Keras 3.x and eliminate deprecation warnings.

#### Key Changes

1. **Model Saving Format**:
   - Updated all model saving operations to use the native Keras format
   - Removed deprecated `save_format` parameter
   - Changed file extensions from `.h5` to `.keras`

2. **Model Loading Compatibility**:
   - Enhanced model loading logic to support both `.keras` and `.h5` formats
   - Added fallback mechanisms to maintain backward compatibility
   - Improved error handling for model loading failures

3. **Feature Engineering**:
   - Improved handling of the `transaction_frequency` feature
   - Enhanced feature consistency across training, evaluation, and inference

4. **Evaluation Improvements**:
   - Fixed model selection logic to properly identify model files
   - Added filtering to exclude non-model files (like JSON results)
   - Improved error reporting and logging

#### Files Modified
- `main.py`: Updated model path handling and evaluation logic
- `src/models/train_model.py`: Updated model saving format
- `src/models/evaluate_model.py`: Enhanced model loading compatibility
- `src/models/save_model_artifacts.py`: Updated artifact saving format
- `src/models/deploy_model.py`: Improved feature handling

#### Performance
- Classification model: 85.6% accuracy
- Autoencoder model: 87.8% accuracy, 60% precision, 45.8% recall for fraud detection

#### Next Steps
- Consider implementing an ensemble approach to combine predictions
- Explore hyperparameter optimization for improved performance
- Conduct feature importance analysis
- API documentation with detailed endpoint descriptions
- Enhanced model deployment with input shape handling
- Fixed port conflict issues in API deployment
- Added test script for API validation
- Added GPU-related command-line arguments to deployment scripts
- Added improved error handling and debugging in API server
- Added GPU-related command-line arguments to streaming fraud detection script
- Added enhanced error handling in streaming fraud detection script with user-friendly messages
- Added compatible test data generation for model monitoring
- Added enhanced model monitoring with better feature mismatch handling
- Added improved recommendations in monitoring reports based on model performance and data drift

### Fixed
- Fixed null reference error in PySpark data preprocessing by replacing 'null' with 'lit(None)'
- Resolved NaN handling in model training and evaluation
- Fixed model file path issues in deployment scripts
- Addressed input feature mismatch between training and deployment
- Fixed GPU parameter handling in model deployment scripts
- Fixed non-numeric column handling in model artifact saving
- Added custom objects support for loading models with custom metrics
- Fixed GPU parameter handling in streaming fraud detection script
- Fixed streaming fraud detection script to provide clear guidance when no socket server is running
- Fixed model monitoring script to handle mismatches between expected and actual features
- Fixed visualization issues in monitoring reports to handle non-numeric metrics
- Fixed metrics calculation in monitoring to provide informative notes when no positive predictions are found

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
