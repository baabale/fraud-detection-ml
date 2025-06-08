#!/bin/bash
# Simple script to test the fraud detection API using curl

# Sample transaction data with all 6 features the model expects
cat > sample_data.json << 'EOF'
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
EOF

# Send request to the API
echo "Sending request to fraud detection API..."
curl -X POST -H "Content-Type: application/json" -d @sample_data.json http://localhost:8080/predict

# Clean up
rm sample_data.json
