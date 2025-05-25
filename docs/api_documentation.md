# Fraud Detection API Documentation

## Overview

The Fraud Detection API provides endpoints for detecting fraudulent transactions using machine learning models. The API supports both classification-based and anomaly-based (autoencoder) fraud detection approaches, as well as an ensemble mode that combines both methods.

## Base URL

```
http://localhost:8080
```

## Authentication

Currently, the API does not require authentication. This is suitable for development and testing purposes only. For production deployment, proper authentication mechanisms should be implemented.

## Endpoints

### Health Check

Check if the API and models are available.

**URL**: `/health`

**Method**: `GET`

**Response**:

```json
{
  "status": "ok",
  "models": {
    "classification": true,
    "autoencoder": false
  }
}
```

The `models` object indicates which models are currently loaded and available.

### Fraud Prediction

Predict whether transactions are fraudulent.

**URL**: `/predict`

**Method**: `POST`

**Content Type**: `application/json`

**Request Body**:

```json
{
  "transactions": [
    {
      "amount": 1000.0,
      "hour": 12,
      "day": 3,
      "feature4": 0.5,
      "feature5": 0.2,
      "feature6": 0.3
    },
    {
      "amount": 10000.0,
      "hour": 2,
      "day": 6,
      "feature4": 0.7,
      "feature5": 0.4,
      "feature6": 0.8
    }
  ],
  "model_type": "classification"
}
```

**Parameters**:

- `transactions` (array, required): List of transactions to evaluate
- `model_type` (string, optional): Type of model to use for prediction. Options:
  - `classification`: Use only the classification model
  - `autoencoder`: Use only the autoencoder model for anomaly detection
  - `ensemble`: Use both models and combine results (default)

**Response**:

```json
{
  "predictions": [
    {
      "fraud_probability": 0.0,
      "is_fraud": false
    },
    {
      "fraud_probability": 0.0,
      "is_fraud": false
    }
  ]
}
```

When using the `classification` model type, each prediction contains:
- `fraud_probability`: Probability of the transaction being fraudulent (0.0 to 1.0)
- `is_fraud`: Boolean indicating whether the transaction is classified as fraudulent

When using the `autoencoder` model type, each prediction contains:
- `anomaly_score`: Anomaly score for the transaction
- `is_fraud`: Boolean indicating whether the transaction is classified as fraudulent based on the anomaly threshold

When using the `ensemble` model type, each prediction contains all of the above fields.

## Error Handling

The API returns appropriate HTTP status codes and error messages in case of failures:

- `400 Bad Request`: Invalid input data
- `500 Internal Server Error`: Server-side error

Error response format:

```json
{
  "error": "Error message details"
}
```

## Example Usage

### Using curl

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {
        "amount": 1000.0,
        "hour": 12,
        "day": 3,
        "feature4": 0.5,
        "feature5": 0.2,
        "feature6": 0.3
      }
    ],
    "model_type": "classification"
  }' \
  http://localhost:8080/predict
```

### Using Python

```python
import requests
import json

url = "http://localhost:8080/predict"

data = {
    "transactions": [
        {
            "amount": 1000.0,
            "hour": 12,
            "day": 3,
            "feature4": 0.5,
            "feature5": 0.2,
            "feature6": 0.3
        }
    ],
    "model_type": "classification"
}

response = requests.post(url, json=data)
predictions = response.json()
print(json.dumps(predictions, indent=2))
```

## Notes

- The API automatically handles missing features by filling them with default values
- For optimal performance, provide all required features in the request
- The API supports batch processing of multiple transactions in a single request
