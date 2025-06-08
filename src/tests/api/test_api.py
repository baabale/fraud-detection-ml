#!/usr/bin/env python
"""
Simple script to test the fraud detection API.
"""
import requests
import json

# API endpoint
url = "http://localhost:8080/predict"

# Sample transaction data
sample_data = {
    "transactions": [
        {
            "amount": 500.0,
            "time_since_last_transaction": 3600,
            "spending_deviation_score": 0.2,
            "velocity_score": 0.3,
            "geo_anomaly_score": 0.1,
            "amount_log": 6.2,
            "hour_of_day": 14,
            "day_of_week": 3,
            "month": 5,
            "year": 2025,
            "fraud_label": 0,
            "velocity_score_norm": 0.3
        },
        {
            "amount": 5000.0,
            "time_since_last_transaction": 1200,
            "spending_deviation_score": 1.5,
            "velocity_score": 0.8,
            "geo_anomaly_score": 0.7,
            "amount_log": 8.5,
            "hour_of_day": 2,
            "day_of_week": 6,
            "month": 5,
            "year": 2025,
            "fraud_label": 0,
            "velocity_score_norm": 0.8
        }
    ],
    "model_type": "both"  # Use both classification and autoencoder models
}

# Send request
try:
    response = requests.post(url, json=sample_data)
    
    # Print response
    print(f"Status Code: {response.status_code}")
    print("Response:")
    print(json.dumps(response.json(), indent=2))
    
except Exception as e:
    print(f"Error: {str(e)}")
