"""
Unit tests for the fraud detection models.
"""

import unittest
import numpy as np
import tensorflow as tf
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from models.fraud_model import create_classification_model, create_autoencoder_model

class TestModels(unittest.TestCase):
    """Test cases for fraud detection models."""
    
    def test_classification_model_creation(self):
        """Test that the classification model can be created with the expected structure."""
        # Create a simple model for testing
        input_dim = 10
        hidden_layers = [8, 4]
        dropout_rate = 0.2
        
        model = create_classification_model(input_dim, hidden_layers, dropout_rate)
        
        # Check model type
        self.assertIsInstance(model, tf.keras.Model)
        
        # Check input shape
        self.assertEqual(model.input_shape, (None, input_dim))
        
        # Check output shape
        self.assertEqual(model.output_shape, (None, 1))
        
        # Check that the model can make predictions
        test_input = np.random.random((5, input_dim))
        predictions = model.predict(test_input)
        
        # Check prediction shape
        self.assertEqual(predictions.shape, (5, 1))
        
        # Check prediction values are between 0 and 1
        self.assertTrue(np.all(predictions >= 0))
        self.assertTrue(np.all(predictions <= 1))
    
    def test_autoencoder_model_creation(self):
        """Test that the autoencoder model can be created with the expected structure."""
        # Create a simple model for testing
        input_dim = 10
        encoding_dim = 2
        hidden_layers = [8, 4]
        
        # Match the function signature: create_autoencoder_model(input_dim, encoding_dim=16, hidden_layers=[64, 32])
        model = create_autoencoder_model(input_dim, encoding_dim, hidden_layers)
        
        # Check model type
        self.assertIsInstance(model, tf.keras.Model)
        
        # Check input shape
        self.assertEqual(model.input_shape, (None, input_dim))
        
        # Check output shape
        self.assertEqual(model.output_shape, (None, input_dim))
        
        # Check that the model can make predictions
        test_input = np.random.random((5, input_dim))
        reconstructions = model.predict(test_input)
        
        # Check reconstruction shape
        self.assertEqual(reconstructions.shape, (5, input_dim))

if __name__ == '__main__':
    unittest.main()
