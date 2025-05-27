"""
Integration tests for the fraud detection API.
"""

import unittest
import json
import os
import sys
import numpy as np
from unittest.mock import patch, MagicMock

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

# Import the Flask app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/api')))
from app import app

class TestFraudDetectionAPI(unittest.TestCase):
    """Test cases for the fraud detection API."""
    
    def setUp(self):
        """Set up the test client."""
        self.app = app.test_client()
        self.app.testing = True
    
    def test_health_endpoint(self):
        """Test the health check endpoint."""
        # Mock the global variables
        with patch('app.classification_model', MagicMock()), \
             patch('app.autoencoder_model', MagicMock()):
            
            response = self.app.get('/health')
            data = json.loads(response.data)
            
            self.assertEqual(response.status_code, 200)
            self.assertEqual(data['status'], 'ok')
    
    def test_health_endpoint_models_not_loaded(self):
        """Test the health check endpoint when models are not loaded."""
        # Mock the global variables to be None
        with patch('app.classification_model', None), \
             patch('app.autoencoder_model', None):
            
            response = self.app.get('/health')
            data = json.loads(response.data)
            
            self.assertEqual(response.status_code, 503)
            self.assertEqual(data['status'], 'error')
            self.assertEqual(data['message'], 'Models not loaded')
    
    def test_predict_endpoint(self):
        """Test the predict endpoint."""
        # Mock the global variables and model predictions
        with patch('app.classification_model') as mock_class_model, \
             patch('app.autoencoder_model') as mock_ae_model, \
             patch('app.feature_names', ['feature1', 'feature2']), \
             patch('app.threshold', 0.5):
            
            # Set up mock predictions
            mock_class_model.predict.return_value = np.array([[0.2]])
            mock_ae_model.predict.return_value = np.array([[0.1, 0.2]])
            
            # Test data
            test_data = {
                'transaction_id': 'test_tx_1',
                'feature1': 0.5,
                'feature2': 0.7
            }
            
            response = self.app.post('/predict', 
                                     data=json.dumps(test_data),
                                     content_type='application/json')
            
            data = json.loads(response.data)
            
            self.assertEqual(response.status_code, 200)
            self.assertEqual(data['status'], 'success')
            self.assertIn('predictions', data)
            self.assertEqual(len(data['predictions']), 1)
            
            prediction = data['predictions'][0]
            self.assertEqual(prediction['transaction_id'], 'test_tx_1')
            self.assertFalse(prediction['is_fraud_classification'])
            self.assertIn('anomaly_score', prediction)
    
    def test_predict_missing_features(self):
        """Test the predict endpoint with missing features."""
        # Mock the global variables
        with patch('app.classification_model', MagicMock()), \
             patch('app.autoencoder_model', MagicMock()), \
             patch('app.feature_names', ['feature1', 'feature2', 'feature3']), \
             patch('app.threshold', 0.5):
            
            # Test data with missing feature
            test_data = {
                'transaction_id': 'test_tx_1',
                'feature1': 0.5,
                'feature2': 0.7
                # Missing feature3
            }
            
            response = self.app.post('/predict', 
                                     data=json.dumps(test_data),
                                     content_type='application/json')
            
            data = json.loads(response.data)
            
            self.assertEqual(response.status_code, 400)
            self.assertEqual(data['status'], 'error')
            self.assertIn('Missing features', data['message'])

if __name__ == '__main__':
    unittest.main()
