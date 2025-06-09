"""
Ensemble models for fraud detection.

This module contains implementations of ensemble models that combine multiple
base models (e.g., classification and autoencoder) for improved fraud detection.
"""

from .ensemble_model import FraudDetectionEnsemble
from .train_ensemble import train_ensemble
from .evaluate_ensemble import evaluate_ensemble

__all__ = [
    'FraudDetectionEnsemble',
    'train_ensemble',
    'evaluate_ensemble'
]
