# Anomaly-Based Fraud Detection in Banking Transactions

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)
![PySpark](https://img.shields.io/badge/PySpark-3.3+-red.svg)
![MLflow](https://img.shields.io/badge/MLflow-2.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A scalable, production-ready system for detecting fraudulent banking transactions using anomaly detection techniques with Spark and deep learning.

## 🔍 Project Overview

This project implements a comprehensive fraud detection system that combines big data processing with sophisticated deep learning models to identify unusual transaction patterns indicating fraud. The system offers both supervised classification and unsupervised anomaly detection approaches.

### Key Features

- **Distributed Data Processing**: Efficiently process millions of transactions using Apache Spark
- **Dual Model Approach**: Classification and autoencoder-based anomaly detection
- **Real-time Detection**: Stream processing for immediate fraud alerts
- **Model Monitoring**: Drift detection and performance tracking
- **Interactive Interface**: User-friendly command menu for all operations
- **Comprehensive Evaluation**: Detailed metrics and visualizations with business impact analysis
- **Advanced Sampling**: Cluster-based SMOTE for better handling of imbalanced fraud data
- **Cost-Sensitive Evaluation**: Financial impact metrics and cost-optimal threshold determination

## 🛠️ Technologies

- **PySpark**: Distributed data processing for large-scale transaction data
- **TensorFlow**: Deep learning models for fraud detection
- **MLflow**: Experiment tracking and model management
- **Jupyter Notebooks**: Interactive analysis and visualization
- **Flask**: Model deployment as a REST API
- **Structured Streaming**: Real-time transaction processing

## 📊 Dataset

The project uses the Kaggle Financial Transactions Dataset for Fraud Detection, containing approximately 5 million simulated banking transactions with attributes including:

- Transaction details (ID, timestamp, amount, sender/receiver accounts)
- Behavioral features (transaction frequency, patterns)
- Metadata (device type, location, payment method)
- Fraud labels (binary classification)

## 📁 Project Structure

```
/data
  /raw           # Original raw data files
  /processed     # Cleaned and transformed data
  /test          # Test data for evaluation

/notebooks       # Jupyter notebooks for analysis
  01_Exploratory_Data_Analysis.ipynb
  02_Model_Training_and_Tuning.ipynb
  03_Model_Evaluation_and_Interpretation.ipynb

/src             # Source code
  /spark_jobs    # PySpark ETL scripts
    load_data.py
    streaming_fraud_detection.py
  /models        # Model definitions and training scripts
    fraud_model.py
    train_model.py
    evaluate_model.py
    deploy_model.py
    save_model_artifacts.py
    model_monitoring.py
  /utils         # Utility modules
    data_utils.py
    generate_test_data.py
  pipeline.py    # Main pipeline script

/results         # Output files
  /models        # Trained models
  /figures       # Visualizations
  /metrics       # Performance metrics
  /deployment    # Deployment artifacts
  /monitoring    # Monitoring results

/docs            # Project documentation
  PROJECT.md
  api_documentation.md  # API usage documentation

main.py          # Interactive command interface
config.yaml      # Configuration parameters
mlflow_config.yaml # MLflow settings
```

## ⚙️ Setup Instructions

### Prerequisites

- Python 3.10+
- Apache Spark 3.3+
- TensorFlow 2.10+
- MLflow 2.0+
- Docker and Docker Compose (for production deployment)

### Development Environment Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/baabale/fraud-detection-ml.git
   cd fraud-detection-ml
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv_fraud
   source venv_fraud/bin/activate  # On Windows: venv_fraud\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the dataset from Kaggle and place it in the `/data/raw` directory.

### Production Environment Setup

1. Start the MLflow tracking server:
   ```bash
   mlflow server --host 0.0.0.0 --port 5000
   ```

2. Run the pipeline with production configuration:
   ```bash
   export CONFIG_PATH=config.production.yaml
   python src/pipeline.py
   ```

3. Alternatively, use Docker for a containerized environment:
   ```bash
   docker-compose up mlflow
   docker-compose up fraud-detection
   ```

## 🚀 Usage

### Running the Pipeline

To run the complete fraud detection pipeline:

```bash
python src/pipeline.py
```

You can specify which steps to run and which model types to use:

```bash
python src/pipeline.py --steps process train evaluate --model-type both
```

Available options:
- `--steps`: Choose from `process`, `train`, `evaluate`, or `all`
- `--model-type`: Choose from `classification`, `autoencoder`, or `both`
- `--config`: Path to configuration file (default: `config.yaml`)

### Individual Components

#### Data Preprocessing
```bash
python src/spark_jobs/load_data.py --input data/raw/sample_dataset.csv --output data/processed/transactions.parquet
```

#### Model Training
```bash
python src/models/train_model.py --data-path data/processed/transactions.parquet --model-type both --experiment-name Fraud_Detection --model-dir results/models
```

#### Model Evaluation
```bash
python src/models/evaluate_model.py --model-path results/models/classification_model_model.h5 --test-data data/processed/transactions.parquet --model-type classification --output-dir results
```

#### Model Evaluation
```bash
python src/models/evaluate_model.py
```

#### Model Deployment
```bash
python src/models/deploy_model.py --model-dir results/deployment --host 0.0.0.0 --port 8080
```

#### API Usage
Once deployed, the fraud detection API can be accessed at `http://localhost:8080`. The API provides the following endpoints:

- `GET /health` - Check API and model status
- `POST /predict` - Submit transactions for fraud detection

For detailed API documentation, see [API Documentation](docs/api_documentation.md).

Example API request using curl:
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"transactions": [{"amount": 1000.0, "hour": 12, "day": 3, "feature4": 0.5, "feature5": 0.2, "feature6": 0.3}], "model_type": "classification"}' \
  http://localhost:8080/predict
```

#### Streaming Fraud Detection
```bash
python src/spark_jobs/streaming_fraud_detection.py
```

## 📈 Results

The system achieves high accuracy in fraud detection through a combination of supervised and unsupervised approaches:

- **Classification model**: High accuracy for known fraud patterns
  - Improved with L2 regularization (0.001) to prevent overfitting
  - Enhanced with cluster-based SMOTE sampling for better preservation of fraud pattern distributions
  - Uses asymmetric focal loss with tuned parameters (gamma=4.0, alpha=0.3) for better fraud detection
  - Includes cost-sensitive evaluation metrics to translate model performance into business value
  - Provides financial impact visualizations to find cost-optimal classification thresholds

- **Autoencoder model**: Effective at detecting novel and unusual transaction patterns
  - Improved with larger encoding dimension (24 instead of 16)
  - Enhanced with L2 regularization for better generalization
  - Optimized threshold selection (90th percentile) for anomaly detection

- **Ensemble approach**: Combines strengths of both models for robust detection

Detailed performance metrics and visualizations are generated in the `/results` directory.

## 🔄 Model Monitoring

The system includes comprehensive monitoring capabilities:

- Data drift detection between training and production data
- Model performance tracking over time
- Automated reporting with recommendations for model updates

## 🚢 Production Deployment

This project is set up for production deployment using Docker and CI/CD pipelines. Follow these steps to deploy the system in a production environment:

### Docker Deployment

1. Build the Docker image:
   ```bash
   docker build -t fraud-detection .
   ```

2. Run the container in different modes:
   
   **Training Pipeline:**
   ```bash
   docker run -v $(pwd)/data:/app/data -v $(pwd)/results:/app/results fraud-detection train
   ```
   
   **API Service:**
   ```bash
   docker run -p 8000:8000 -v $(pwd)/results:/app/results fraud-detection api
   ```
   
   **Streaming Service:**
   ```bash
   docker run -v $(pwd)/data:/app/data -v $(pwd)/results:/app/results fraud-detection stream
   ```
   
   **Monitoring Service:**
   ```bash
   docker run -v $(pwd)/results:/app/results fraud-detection monitor
   ```

3. Using Docker Compose (recommended for production):
   ```bash
   docker-compose up -d
   ```
   This will start all services defined in the `docker-compose.yml` file.

### CI/CD Pipeline

The project includes a GitHub Actions workflow for continuous integration and deployment:

1. **Automated Testing**: Runs unit and integration tests on every push and pull request
2. **Docker Build**: Builds and pushes the Docker image to GitHub Container Registry
3. **Deployment**: Automatically deploys to production when changes are merged to the main branch

To set up the CI/CD pipeline:

1. Configure the necessary secrets in your GitHub repository:
   - `DEPLOY_HOST`: Hostname of your deployment server
   - `DEPLOY_USERNAME`: Username for SSH access
   - `DEPLOY_KEY`: SSH private key for deployment

2. Uncomment the deployment job in `.github/workflows/ci-cd.yml` and update the deployment path.

### Production Configuration

The system uses separate configuration files for development and production:

- `config.yaml`: Development configuration
- `config.production.yaml`: Production configuration

To use the production configuration, set the `CONFIG_PATH` environment variable:

```bash
export CONFIG_PATH=config.production.yaml
```

The Docker containers automatically use the production configuration.

### Monitoring and Logging

In production, the system provides:

1. **Prometheus Metrics**: Available at the `/metrics` endpoint
2. **Centralized Logging**: All logs are stored in `/app/logs/fraud_detection.log`
3. **Model Drift Detection**: Automatically monitors for data and model drift

### Scaling Considerations

For high-throughput environments:

1. **Horizontal Scaling**: Deploy multiple API containers behind a load balancer
2. **Spark Cluster**: Configure the system to connect to an external Spark cluster
3. **Distributed Training**: Utilize distributed TensorFlow for model training on large datasets

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Kaggle for providing the financial transactions dataset
- The Apache Spark and TensorFlow communities for their excellent tools
- Contributors to the various open-source libraries used in this project
