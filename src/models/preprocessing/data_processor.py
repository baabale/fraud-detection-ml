"""
Unified data processor for fraud detection models.
"""
import os
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

# Import sampling techniques if available
try:
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
    SAMPLING_AVAILABLE = True
except ImportError:
    SAMPLING_AVAILABLE = False

# Set up logging
logger = logging.getLogger(__name__)

class FraudDataProcessor:
    """
    Unified data processor for fraud detection models.
    
    This class provides consistent preprocessing functionality for both
    classification and autoencoder models, ensuring that the ensemble
    model uses the same preprocessing steps for both components.
    """
    
    def __init__(self, 
                 classification_features: List[str] = None,
                 autoencoder_features: List[str] = None,
                 target_column: str = 'is_fraud',
                 random_state: int = 42):
        """
        Initialize the data processor.
        
        Args:
            classification_features: List of features for classification model
            autoencoder_features: List of features for autoencoder model
            target_column: Name of target column
            random_state: Random seed for reproducibility
        """
        self.classification_features = classification_features
        self.autoencoder_features = autoencoder_features
        self.target_column = target_column
        self.random_state = random_state
        self.classification_scaler = StandardScaler()
        self.autoencoder_scaler = StandardScaler()
        
    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load data from file with optimized settings.
        
        Args:
            data_path: Path to the data file
            
        Returns:
            Loaded DataFrame
        """
        logger.info(f"Loading data from {data_path}")
        start_time = time.time()
        
        try:
            if data_path.endswith('.parquet'):
                # Determine which columns to load
                usecols = list(set(
                    (self.classification_features or []) + 
                    (self.autoencoder_features or []) + 
                    [self.target_column]
                ))
                # Use memory mapping for large files
                df = pd.read_parquet(data_path, columns=usecols)
            elif data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
            else:
                raise ValueError(f"Unsupported file format: {data_path}")
                
            logger.info(f"Loaded {len(df):,} rows in {time.time() - start_time:.2f} seconds")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
    def preprocess_features(self, df: pd.DataFrame, feature_list: List[str]) -> np.ndarray:
        """
        Preprocess features for a specific model.
        
        Args:
            df: Input DataFrame
            feature_list: List of features to process
            
        Returns:
            Preprocessed feature array
        """
        # Select features
        X = df[feature_list].copy()
        
        # Convert object columns to numeric
        object_cols = X.select_dtypes(include=['object']).columns
        for col in object_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Handle missing values with mean imputation
        nan_counts = X.isna().sum()
        if nan_counts.sum() > 0:
            logger.info(f"Filling {nan_counts.sum()} NaN values with column means")
            X = X.fillna(X.mean())
        
        # Add derived features if needed
        if 'transaction_frequency' not in X.columns and 'transaction_frequency' in feature_list:
            if 'time_since_last_transaction' in X.columns:
                # Convert to transactions per day (86400 seconds in a day)
                X['transaction_frequency'] = np.where(
                    X['time_since_last_transaction'] > 0,
                    86400 / X['time_since_last_transaction'],
                    1.0  # Default value for first transaction
                )
                logger.info("Created 'transaction_frequency' feature from 'time_since_last_transaction'")
            else:
                # If no time data available, use a default value
                X['transaction_frequency'] = 1.0
                logger.info("Added 'transaction_frequency' with default value of 1.0")
        
        # Final check for NaNs and convert to numpy
        if X.isna().any().any():
            logger.warning("NaN values still present after filling. Replacing with zeros.")
            X = X.fillna(0)
            
        return X.values.astype(np.float32)
    
    def process_data(self, 
                    data_path: str,
                    val_split: float = 0.2, 
                    test_split: float = 0.0,
                    apply_sampling: bool = False,
                    sampling_technique: str = 'none',
                    sampling_ratio: float = 0.5,
                    k_neighbors: int = 5) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Process data for ensemble model training.
        
        Args:
            data_path: Path to the data file
            val_split: Fraction of data to use for validation
            test_split: Fraction of data to use for testing
            apply_sampling: Whether to apply sampling techniques
            sampling_technique: Sampling technique to use ('smote', 'adasyn', 'borderline_smote')
            sampling_ratio: Desired ratio of minority to majority class
            k_neighbors: Number of neighbors for sampling techniques
            
        Returns:
            Dictionary containing processed data for both models
        """
        # Load data
        df = self.load_data(data_path)
        
        # Extract target
        y = df[self.target_column].values
        
        # Process features for each model
        results = {}
        
        if self.classification_features:
            # Preprocess classification features
            X_clf = self.preprocess_features(df, self.classification_features)
            
            # Split data
            clf_splits = self._split_data(X_clf, y, val_split, test_split)
            
            # Scale features
            clf_splits['X_train'] = self.classification_scaler.fit_transform(clf_splits['X_train'])
            if len(clf_splits['X_val']) > 0:
                clf_splits['X_val'] = self.classification_scaler.transform(clf_splits['X_val'])
            if len(clf_splits['X_test']) > 0:
                clf_splits['X_test'] = self.classification_scaler.transform(clf_splits['X_test'])
                
            # Apply sampling if requested
            if apply_sampling and sampling_technique != 'none':
                clf_splits = self._apply_sampling(
                    clf_splits, 
                    sampling_technique, 
                    sampling_ratio,
                    k_neighbors
                )
                
            results['classification'] = clf_splits
            
        if self.autoencoder_features:
            # Preprocess autoencoder features
            X_ae = self.preprocess_features(df, self.autoencoder_features)
            
            # Split data
            ae_splits = self._split_data(X_ae, y, val_split, test_split)
            
            # Scale features
            ae_splits['X_train'] = self.autoencoder_scaler.fit_transform(ae_splits['X_train'])
            if len(ae_splits['X_val']) > 0:
                ae_splits['X_val'] = self.autoencoder_scaler.transform(ae_splits['X_val'])
            if len(ae_splits['X_test']) > 0:
                ae_splits['X_test'] = self.autoencoder_scaler.transform(ae_splits['X_test'])
                
            results['autoencoder'] = ae_splits
            
        return results
        
    def _split_data(self, X: np.ndarray, y: np.ndarray, 
                   val_split: float, test_split: float) -> Dict[str, np.ndarray]:
        """
        Split data into train/val/test sets.
        
        Args:
            X: Feature array
            y: Target array
            val_split: Fraction of data to use for validation
            test_split: Fraction of data to use for testing
            
        Returns:
            Dictionary containing split data
        """
        result = {}
        
        # Split into train and test
        if test_split <= 0:
            X_train, X_test = X, np.array([])
            y_train, y_test = y, np.array([])
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_split, random_state=self.random_state, 
                stratify=y if len(np.unique(y)) > 1 else None
            )
            
        # Split into train and validation
        if val_split > 0 and len(X_train) > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=val_split, random_state=self.random_state,
                stratify=y_train if len(np.unique(y_train)) > 1 else None
            )
        else:
            X_val, y_val = np.array([]), np.array([])
            
        result['X_train'] = X_train
        result['X_val'] = X_val
        result['X_test'] = X_test
        result['y_train'] = y_train
        result['y_val'] = y_val
        result['y_test'] = y_test
        
        return result
        
    def _apply_sampling(self, data_splits: Dict[str, np.ndarray], 
                       technique: str, ratio: float, k_neighbors: int) -> Dict[str, np.ndarray]:
        """
        Apply sampling technique to training data.
        
        Args:
            data_splits: Dictionary containing split data
            technique: Sampling technique to use
            ratio: Desired ratio of minority to majority class
            k_neighbors: Number of neighbors for sampling techniques
            
        Returns:
            Updated data splits with resampled training data
        """
        if not SAMPLING_AVAILABLE:
            logger.warning("Sampling techniques not available. Install imbalanced-learn package.")
            return data_splits
            
        X_train = data_splits['X_train']
        y_train = data_splits['y_train']
        
        try:
            # Print class distribution before sampling
            class_counts_before = np.bincount(y_train.astype(int))
            logger.info(f"Class distribution before sampling: {class_counts_before}")
            if len(class_counts_before) > 1:
                fraud_ratio_before = class_counts_before[1] / len(y_train) * 100
                logger.info(f"Fraud ratio before sampling: {fraud_ratio_before:.2f}%")
                
            # Apply the sampling technique
            if technique.lower() == 'smote':
                sampler = SMOTE(sampling_strategy=ratio, k_neighbors=k_neighbors, random_state=self.random_state)
            elif technique.lower() == 'adasyn':
                sampler = ADASYN(sampling_strategy=ratio, n_neighbors=k_neighbors, random_state=self.random_state)
            elif technique.lower() == 'borderline_smote':
                sampler = BorderlineSMOTE(sampling_strategy=ratio, k_neighbors=k_neighbors, random_state=self.random_state)
            else:
                logger.warning(f"Unknown sampling technique '{technique}'. Using original data.")
                return data_splits
                
            # Apply the sampling
            X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)
            
            # Update the training data
            data_splits['X_train'] = X_train_resampled
            data_splits['y_train'] = y_train_resampled
            
            # Print class distribution after sampling
            class_counts_after = np.bincount(y_train_resampled.astype(int))
            logger.info(f"Class distribution after sampling: {class_counts_after}")
            if len(class_counts_after) > 1:
                fraud_ratio_after = class_counts_after[1] / len(y_train_resampled) * 100
                logger.info(f"Fraud ratio after sampling: {fraud_ratio_after:.2f}%")
                logger.info(f"Sampling increased fraud ratio from {fraud_ratio_before:.2f}% to {fraud_ratio_after:.2f}%")
                
        except Exception as e:
            logger.error(f"Error applying sampling technique: {str(e)}")
            logger.info("Using original data without sampling.")
            
        return data_splits
        
    def save(self, output_dir: str) -> None:
        """
        Save processor state including scalers.
        
        Args:
            output_dir: Directory to save processor state
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save scalers
        if hasattr(self, 'classification_scaler'):
            joblib.dump(self.classification_scaler, 
                       os.path.join(output_dir, 'classification_scaler.joblib'))
            
        if hasattr(self, 'autoencoder_scaler'):
            joblib.dump(self.autoencoder_scaler,
                       os.path.join(output_dir, 'autoencoder_scaler.joblib'))
            
        # Save feature lists
        metadata = {
            'classification_features': self.classification_features,
            'autoencoder_features': self.autoencoder_features,
            'target_column': self.target_column,
            'random_state': self.random_state
        }
        
        import json
        with open(os.path.join(output_dir, 'data_processor_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
            
    @classmethod
    def load(cls, input_dir: str) -> 'FraudDataProcessor':
        """
        Load saved processor state.
        
        Args:
            input_dir: Directory containing saved processor state
            
        Returns:
            Loaded FraudDataProcessor instance
        """
        import json
        
        # Load metadata
        try:
            with open(os.path.join(input_dir, 'data_processor_metadata.json'), 'r') as f:
                metadata = json.load(f)
        except FileNotFoundError:
            metadata = {}
            
        # Create processor with metadata
        processor = cls(
            classification_features=metadata.get('classification_features'),
            autoencoder_features=metadata.get('autoencoder_features'),
            target_column=metadata.get('target_column', 'is_fraud'),
            random_state=metadata.get('random_state', 42)
        )
        
        # Load scalers
        clf_scaler_path = os.path.join(input_dir, 'classification_scaler.joblib')
        ae_scaler_path = os.path.join(input_dir, 'autoencoder_scaler.joblib')
        
        if os.path.exists(clf_scaler_path):
            processor.classification_scaler = joblib.load(clf_scaler_path)
            
        if os.path.exists(ae_scaler_path):
            processor.autoencoder_scaler = joblib.load(ae_scaler_path)
            
        return processor
