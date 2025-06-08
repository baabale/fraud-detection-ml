# Advanced Fraud Detection Techniques

This document provides an overview of the advanced techniques implemented to improve fraud detection recall in our system.

## Problem: Low Recall for Fraud Detection

Traditional machine learning approaches often struggle with imbalanced datasets like fraud detection, where:
- Fraudulent transactions are rare (minority class)
- The cost of missing fraud (false negatives) is much higher than false alarms (false positives)
- Standard metrics like accuracy can be misleading

Our implementation addresses these challenges with two key approaches:

1. **Advanced Sampling Techniques** - To balance the training data
2. **Custom Loss Functions** - To focus the model on correctly identifying fraud cases

## 1. Advanced Sampling Techniques

We've implemented several sampling techniques to address class imbalance:

### Available Techniques

| Technique | Description | Best For |
|-----------|-------------|----------|
| SMOTE | Synthetic Minority Over-sampling Technique - Creates synthetic samples of the minority class | General imbalanced datasets |
| ADASYN | Adaptive Synthetic Sampling - Similar to SMOTE but focuses on difficult-to-learn examples | Datasets with complex decision boundaries |
| Borderline SMOTE | Focuses on generating samples near the decision boundary | When you want to refine the decision boundary |

### Configuration

You can configure these techniques in `config.yaml`:

```yaml
# Advanced sampling parameters
sampling:
  technique: "smote"  # Options: none, smote, adasyn, borderline_smote
  ratio: 0.7          # Ratio of minority to majority class after sampling
  k_neighbors: 5      # Number of neighbors for SMOTE and variants
```

### Command-line Usage

```bash
python src/models/train_model.py --sampling-technique smote --sampling-ratio 0.7 --k-neighbors 5
```

## 2. Custom Loss Functions

We've implemented several custom loss functions designed specifically for imbalanced classification:

### Available Loss Functions

| Loss Function | Description | Best For |
|---------------|-------------|----------|
| Focal Loss | Down-weights well-classified examples, focusing on hard examples | General imbalanced datasets |
| Weighted Focal Loss | Focal loss with class weights | Highly imbalanced datasets |
| Asymmetric Focal Loss | Different focusing parameters for positive/negative classes | When false negatives are more costly |
| Adaptive Focal Loss | Dynamically adjusts focusing parameters | Complex datasets with varying difficulty |

### Configuration

You can configure these loss functions in `config.yaml`:

```yaml
# Custom loss function parameters
loss_function:
  name: "focal"       # Options: binary_crossentropy, focal, weighted_focal, asymmetric_focal, adaptive_focal
  focal_gamma: 2.0    # Focusing parameter for focal loss
  focal_alpha: 0.25   # Alpha parameter for focal loss
  class_weight_ratio: 10.0  # Weight ratio for positive class (fraud) to negative class
```

### Command-line Usage

```bash
python src/models/train_model.py --loss-function focal --focal-gamma 2.0 --focal-alpha 0.25 --class-weight-ratio 10.0
```

## Results Analysis

Our implementation shows significant improvements in fraud detection recall:

### Classification Model Results

Looking at the terminal output, we can see that the classification model with SMOTE sampling and focal loss achieved:
- Perfect recall (1.0) for fraud cases in the validation set
- Improved recall for fraud cases in the test set compared to baseline

### Autoencoder Model Results

The autoencoder model showed:
- Good precision for normal transactions (0.70)
- Lower recall for fraud cases, which is expected for anomaly detection approaches

## Recommendations for Optimal Performance

Based on our experiments, we recommend:

1. **For highest fraud recall**: Use SMOTE with a sampling ratio of 0.7 and focal loss with gamma=2.0
2. **For balanced precision/recall**: Use Borderline SMOTE with weighted focal loss
3. **For production deployment**: Combine both classification and autoencoder models in an ensemble

## Further Improvements

Potential areas for further improvement:

1. Experiment with different sampling ratios (0.5-0.8)
2. Fine-tune focal loss parameters (gamma between 1.0-3.0)
3. Implement ensemble methods combining multiple models
4. Add feature importance analysis to identify the most predictive features for fraud

## Usage in the Pipeline

The advanced sampling and custom loss functions are fully integrated into the pipeline and can be used by:

1. Modifying the `config.yaml` file
2. Running the complete pipeline with `python main.py` and selecting option 1
3. Or running specific components with command-line arguments

## References

- Focal Loss: [Lin et al., "Focal Loss for Dense Object Detection"](https://arxiv.org/abs/1708.02002)
- SMOTE: [Chawla et al., "SMOTE: Synthetic Minority Over-sampling Technique"](https://arxiv.org/abs/1106.1813)
- ADASYN: [He et al., "ADASYN: Adaptive Synthetic Sampling Approach for Imbalanced Learning"](https://ieeexplore.ieee.org/document/4633969)
