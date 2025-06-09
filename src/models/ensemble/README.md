# Fraud Detection Ensemble Model

## Overview

This module implements an ensemble model that combines a classification model and an autoencoder for fraud detection. The ensemble leverages the strengths of both approaches:

1. **Classification Model**: Provides explicit fraud probability scores based on supervised learning
2. **Autoencoder Model**: Identifies anomalies through reconstruction errors

## Key Components

### Unified Data Processor

The ensemble now uses a unified data processor (`FraudDataProcessor`) that ensures consistent preprocessing across:
- Training
- Validation
- Inference

This processor handles:
- Data loading with multi-GPU optimizations
- Feature-specific preprocessing
- Missing value imputation
- Train/validation/test splitting with stratification
- Feature scaling with separate scalers for classification and autoencoder
- Advanced sampling techniques (SMOTE, ADASYN, BorderlineSMOTE)

### Ensemble Model

The `FraudDetectionEnsemble` class combines predictions from both models with configurable weights:
- Classification weight (default: 0.7)
- Autoencoder weight (default: 0.3)

The ensemble provides:
- Binary predictions using a configurable threshold
- Probability scores for fraud detection
- Automatic threshold tuning based on validation data

## Usage

### Training the Ensemble

```bash
python src/models/ensemble/train_ensemble.py \
  --data-path /path/to/data.parquet \
  --classification-model /path/to/classification_model.keras \
  --autoencoder-model /path/to/autoencoder_model.keras \
  --output-dir /path/to/output \
  --classification-weight 0.7 \
  --val-split 0.2 \
  --apply-sampling \
  --sampling-technique smote \
  --sampling-ratio 0.5
```

### Key Parameters

- `--classification-weight`: Weight for classification model (0-1)
- `--threshold`: Decision threshold (if not specified, will find optimal threshold)
- `--target-recall`: Target recall to achieve (overrides threshold)
- `--target-precision`: Target precision to achieve (overrides threshold)
- `--apply-sampling`: Enable advanced sampling techniques
- `--sampling-technique`: Sampling technique to use (smote, adasyn, borderline_smote)
- `--sampling-ratio`: Desired ratio of minority to majority class

## Performance Considerations

Based on historical model performance:

1. **Class Imbalance**: The model is sensitive to class imbalance. Use sampling techniques cautiously.

2. **Feature Importance**: Temporal and amount-related features consistently show highest importance:
   - time_since_prev_tx
   - transaction_frequency
   - amount_deviation_from_avg
   - amount_ratio

3. **Balanced Configuration**: Current optimal settings:
   - Classification weight: 0.7
   - Autoencoder weight: 0.3
   - Threshold: Optimized for F1 score (typically around 0.5)
   - Sampling: Disabled or light SMOTE (ratio 0.3-0.5)

4. **Evaluation**: Use the 90th percentile threshold for autoencoder reconstruction errors
