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
            "amount": 1000.0,
            "hour": 12,
            "day": 3,
            "spending_deviation_score": 0.5,
            "geo_anomaly_score": 0.2,
            "amount_log": 6.9,
            "velocity_score_norm": 0.3
        },
        {
            "amount": 10000.0,
            "hour": 2,
            "day": 6,
            "spending_deviation_score": 2.5,
            "geo_anomaly_score": 0.9,
            "amount_log": 9.2,
            "velocity_score_norm": 0.8
        }
    ],
    "model_type": "classification"  # Use classification model only
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
