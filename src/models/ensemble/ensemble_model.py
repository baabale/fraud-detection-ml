"""
Ensemble model combining classification and autoencoder models for fraud detection.
"""
import os
import numpy as np
import tensorflow as tf
from typing import Dict, Tuple, Optional, Union, List
from pathlib import Path
import joblib
import json

# Import the data processor
try:
    from src.models.preprocessing.data_processor import FraudDataProcessor
    DATA_PROCESSOR_AVAILABLE = True
except ImportError:
    DATA_PROCESSOR_AVAILABLE = False

class FraudDetectionEnsemble:
    """
    Ensemble model that combines predictions from a classification model and an autoencoder.
    
    The ensemble uses a weighted combination of both models' predictions to make
    the final fraud detection decision.
    """
    
    def __init__(
        self, 
        classification_model: tf.keras.Model = None,
        autoencoder_model: tf.keras.Model = None,
        threshold: float = 0.5,
        classification_weight: float = 0.7,
        autoencoder_weight: float = 0.3,
        data_processor = None,
        classification_features: List[str] = None,
        autoencoder_features: List[str] = None
    ):
        """
        Initialize the ensemble model.
        
        Args:
            classification_model: Trained classification model
            autoencoder_model: Trained autoencoder model
            threshold: Decision threshold for final prediction
            classification_weight: Weight for classification model predictions (0-1)
            autoencoder_weight: Weight for autoencoder predictions (0-1)
        """
        self.classification_model = classification_model
        self.autoencoder_model = autoencoder_model
        self.threshold = threshold
        self.classification_weight = classification_weight
        self.autoencoder_weight = autoencoder_weight
        self.scaler = None
        self.feature_names = None
        self.classification_features = classification_features
        self.autoencoder_features = autoencoder_features
        
        # Initialize data processor if available
        self.data_processor = data_processor
        if self.data_processor is None and DATA_PROCESSOR_AVAILABLE:
            self.data_processor = FraudDataProcessor(
                classification_features=classification_features,
                autoencoder_features=autoencoder_features
            )
        
        # Ensure weights sum to 1
        total_weight = classification_weight + autoencoder_weight
        if total_weight != 1.0:
            self.classification_weight /= total_weight
            self.autoencoder_weight /= total_weight
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the ensemble model.
        
        Args:
            X: Input features
            
        Returns:
            Array of binary predictions (0 or 1)
        """
        # Get classification model predictions
        if self.classification_model is not None:
            y_pred_clf = self.classification_model.predict(X_clf, verbose=0)
            y_pred_clf = y_pred_clf.reshape(-1, 1)
        else:
            y_pred_clf = np.zeros((len(X), 1))
        
        # Get autoencoder reconstruction errors
        if self.autoencoder_model is not None:
            # Apply preprocessing if data processor is available
            if self.data_processor and hasattr(self.data_processor, 'autoencoder_scaler'):
                X_ae_processed = self.data_processor.autoencoder_scaler.transform(X)
                X_ae = self._select_features_for_autoencoder(X_ae_processed)
            else:
                # Select features for autoencoder using the default method
                X_ae = self._select_features_for_autoencoder(X)
            
            # Get reconstruction errors
            X_reconstructed = self.autoencoder_model.predict(X_ae, verbose=0)
            reconstruction_errors = np.mean(np.square(X_ae - X_reconstructed), axis=1)
            
            # Normalize reconstruction errors to [0, 1]
            if hasattr(self, 'recon_scaler_'):
                reconstruction_errors = self.recon_scaler_.transform(reconstruction_errors.reshape(-1, 1))
            
            # Convert to probability-like scores (higher error = higher probability of fraud)
            y_pred_ae = reconstruction_errors.reshape(-1, 1)
        else:
            y_pred_ae = np.zeros((len(X), 1))
        
        # Combine predictions using weights
        combined_scores = (self.classification_weight * y_pred_clf + 
                         self.autoencoder_weight * y_pred_ae)
        
        # Apply threshold
        return (combined_scores >= self.threshold).astype(int)
    
    def predict_proba_raw(self, X):
        """
        Generate probability predictions for input samples without using the data processor.
        This is a fallback method when the data processor is not available or has issues.
        
        Args:
            X: Input features (already preprocessed)
            
        Returns:
            Array of probability scores for fraud class
        """
        # Get classification probabilities
        if self.classification_model is not None:
            clf_probs = self.classification_model.predict(X, verbose=0)
            if len(clf_probs.shape) > 1 and clf_probs.shape[1] > 1:
                # Multi-class output, take the fraud class probability
                clf_probs = clf_probs[:, 1]
        else:
            # No classification model, use zeros
            clf_probs = np.zeros(len(X))
            
        # Get autoencoder reconstruction errors
        if self.autoencoder_model is not None:
            # Get reconstructions
            reconstructed = self.autoencoder_model.predict(X, verbose=0)
            
            # Calculate reconstruction errors
            reconstruction_errors = np.mean(np.square(X - reconstructed), axis=1)
            
            # Scale errors to [0, 1] range using the fitted scaler
            if hasattr(self, 'reconstruction_error_scaler') and self.reconstruction_error_scaler is not None:
                ae_probs = self.reconstruction_error_scaler.transform(reconstruction_errors.reshape(-1, 1)).ravel()
            else:
                # If no scaler, use min-max scaling on the fly
                min_error = np.min(reconstruction_errors)
                max_error = np.max(reconstruction_errors)
                ae_probs = (reconstruction_errors - min_error) / (max_error - min_error + 1e-10)
        else:
            # No autoencoder model, use zeros
            ae_probs = np.zeros(len(X))
            
        # Combine probabilities using weights
        combined_probs = (self.classification_weight * clf_probs + 
                          self.autoencoder_weight * ae_probs)
        
        return combined_probs
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability scores from the ensemble model.
        
        Args:
            X: Input features
            
        Returns:
            Array of probability scores [0, 1]
        """
        # Ensure X is numpy array and has the right shape
        X = np.asarray(X)
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        # Apply preprocessing if data processor is available
        X_clf = X
        if self.data_processor and hasattr(self.data_processor, 'classification_scaler'):
            X_clf = self.data_processor.classification_scaler.transform(X)
            
        # Get classification model predictions
        if self.classification_model is not None:
            y_pred_clf = self.classification_model.predict(X, verbose=0)
            # Ensure we have the right shape (n_samples, 1)
            y_pred_clf = np.asarray(y_pred_clf).reshape(-1, 1)
        else:
            y_pred_clf = np.zeros((X.shape[0], 1))
        
        # Get autoencoder reconstruction errors if autoencoder is available
        if self.autoencoder_model is not None:
            try:
                # Select features for autoencoder
                X_ae = self._select_features_for_autoencoder(X)
                
                # Get reconstruction errors
                X_reconstructed = self.autoencoder_model.predict(X_ae, verbose=0)
                reconstruction_errors = np.mean(np.square(X_ae - X_reconstructed), axis=1)
                
                # Normalize reconstruction errors to [0, 1]
                if hasattr(self, 'recon_scaler_'):
                    reconstruction_errors = self.recon_scaler_.transform(reconstruction_errors.reshape(-1, 1)).flatten()
                
                # Convert to probability-like scores (higher error = higher probability of fraud)
                y_pred_ae = reconstruction_errors.reshape(-1, 1)
            except Exception as e:
                logger.warning(f"Error in autoencoder prediction: {str(e)}")
                y_pred_ae = np.zeros((X.shape[0], 1))
        else:
            y_pred_ae = np.zeros((X.shape[0], 1))
        
        # Ensure both predictions have the same shape
        if y_pred_clf.shape[0] != y_pred_ae.shape[0]:
            min_samples = min(y_pred_clf.shape[0], y_pred_ae.shape[0])
            y_pred_clf = y_pred_clf[:min_samples]
            y_pred_ae = y_pred_ae[:min_samples]
        
        # Combine predictions using weights and ensure output is in [0, 1] range
        combined = (self.classification_weight * y_pred_clf + 
                   self.autoencoder_weight * y_pred_ae)
        return np.clip(combined, 0, 1)
    
    def _select_features_for_autoencoder(self, X):
        """
        Select features for the autoencoder model.
        
        If autoencoder_features is provided, use those indices.
        Otherwise, default to the first 22 features.
        
        Args:
            X: Input features (numpy array)
            
        Returns:
            Selected features for the autoencoder
        """
        if self.autoencoder_features is not None and hasattr(self, 'feature_indices_map'):
            # Use the feature indices map to select features
            indices = [self.feature_indices_map.get(feat, i) 
                      for i, feat in enumerate(self.autoencoder_features) 
                      if feat in self.feature_indices_map]
            if indices:
                return X[:, indices]
                
        # Default to first 22 features if no specific features defined
        if X.shape[1] > 22:
            return X[:, :22]
        return X
        
    def fit_reconstruction_scaler(self, X):
        """
        Fit the reconstruction error scaler on the validation data.
        
        Args:
            X: Input features (numpy array)
        """
        # Select features for autoencoder
        X_ae = self._select_features_for_autoencoder(X)
        
        # Get reconstructions
        X_reconstructed = self.autoencoder_model.predict(X_ae, verbose=0)
        
        # Calculate reconstruction errors
        reconstruction_errors = np.mean(np.power(X_ae - X_reconstructed, 2), axis=1)
        
        # Fit the scaler
        from sklearn.preprocessing import MinMaxScaler
        self.recon_scaler_ = MinMaxScaler()
        self.recon_scaler_.fit(reconstruction_errors.reshape(-1, 1))
    
    def save(self, directory: str):
        """
        Save the ensemble model to disk.
        
        Args:
            directory: Directory to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Save models
        if self.classification_model is not None:
            self.classification_model.save(os.path.join(directory, 'classification_model.keras'))
        if self.autoencoder_model is not None:
            self.autoencoder_model.save(os.path.join(directory, 'autoencoder_model.keras'))
        
        # Save scaler and other attributes
        if hasattr(self, 'recon_scaler_'):
            joblib.dump(self.recon_scaler_, os.path.join(directory, 'recon_scaler.joblib'))
        
        # Save data processor if available
        if self.data_processor is not None:
            self.data_processor.save(os.path.join(directory, 'data_processor'))
        
        # Save feature lists
        feature_data = {
            'classification_features': self.classification_features,
            'autoencoder_features': self.autoencoder_features,
            'feature_indices_map': getattr(self, 'feature_indices_map', {})
        }
        
        with open(os.path.join(directory, 'feature_data.json'), 'w') as f:
            json.dump(feature_data, f)
        
        # Save metadata
        metadata = {
            'threshold': self.threshold,
            'classification_weight': self.classification_weight,
            'autoencoder_weight': self.autoencoder_weight,
        }
        
        with open(os.path.join(directory, 'ensemble_metadata.json'), 'w') as f:
            json.dump(metadata, f)
    
    @classmethod
    def load(cls, directory: str):
        """
        Load a saved ensemble model from disk.
        
        Args:
            directory: Directory containing the saved model
            
        Returns:
            Loaded ensemble model
        """
        # Load models
        classification_model = None
        autoencoder_model = None
        
        if os.path.exists(os.path.join(directory, 'classification_model.keras')):
            classification_model = tf.keras.models.load_model(
                os.path.join(directory, 'classification_model.keras'),
                compile=False
            )
        
        if os.path.exists(os.path.join(directory, 'autoencoder_model.keras')):
            autoencoder_model = tf.keras.models.load_model(
                os.path.join(directory, 'autoencoder_model.keras'),
                compile=False
            )
        
        # Load metadata
        with open(os.path.join(directory, 'ensemble_metadata.json'), 'r') as f:
            metadata = json.load(f)
            
        # Load feature data if it exists
        classification_features = None
        autoencoder_features = None
        feature_indices_map = {}
        
        feature_data_path = os.path.join(directory, 'feature_data.json')
        if os.path.exists(feature_data_path):
            with open(feature_data_path, 'r') as f:
                feature_data = json.load(f)
                classification_features = feature_data.get('classification_features')
                autoencoder_features = feature_data.get('autoencoder_features')
                feature_indices_map = feature_data.get('feature_indices_map', {})
        
        # Load data processor if available
        data_processor = None
        data_processor_dir = os.path.join(directory, 'data_processor')
        if os.path.exists(data_processor_dir) and DATA_PROCESSOR_AVAILABLE:
            try:
                data_processor = FraudDataProcessor.load(data_processor_dir)
            except Exception as e:
                print(f"Warning: Could not load data processor: {str(e)}")
        
        # Create ensemble instance
        ensemble = cls(
            classification_model=classification_model,
            autoencoder_model=autoencoder_model,
            threshold=metadata['threshold'],
            classification_weight=metadata['classification_weight'],
            autoencoder_weight=metadata['autoencoder_weight'],
            data_processor=data_processor,
            classification_features=classification_features,
            autoencoder_features=autoencoder_features
        )
        
        # Set feature indices map
        if feature_indices_map:
            ensemble.feature_indices_map = feature_indices_map
        
        # Load scaler if it exists
        scaler_path = os.path.join(directory, 'recon_scaler.joblib')
        if os.path.exists(scaler_path):
            ensemble.recon_scaler_ = joblib.load(scaler_path)
        
        return ensemble
