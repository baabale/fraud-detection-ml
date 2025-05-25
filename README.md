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
- **Comprehensive Evaluation**: Detailed metrics and visualizations

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

main.py          # Interactive command interface
config.yaml      # Configuration parameters
mlflow_config.yaml # MLflow settings
```

## ⚙️ Setup Instructions

### Prerequisites

- Python 3.8+
- Apache Spark 3.x
- TensorFlow 2.10+
- MLflow 2.0+
- Jupyter Notebook/Lab

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/baabale/fraud-detection-ml.git
   cd fraud-detection-ml
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the dataset from Kaggle and place it in the `/data/raw` directory.

## 🚀 Usage

The project includes an interactive interface for all operations. Simply run:

```bash
python main.py
```

This will display a menu with the following options:

1. **Run the complete pipeline**
2. **Process the data**
3. **Train models**
4. **Evaluate models**
5. **Deploy models as API**
6. **Run streaming fraud detection**
7. **Generate synthetic test data**
8. **Monitor model performance**
9. **Launch Jupyter notebooks**

### Manual Operation

You can also run individual components directly:

#### Data Preprocessing
```bash
python src/spark_jobs/load_data.py
```

#### Model Training
```bash
python src/models/train_model.py
```

#### Model Evaluation
```bash
python src/models/evaluate_model.py
```

#### Model Deployment
```bash
python src/models/deploy_model.py
```

#### Streaming Fraud Detection
```bash
python src/spark_jobs/streaming_fraud_detection.py
```

## 📈 Results

The system achieves high accuracy in fraud detection through a combination of supervised and unsupervised approaches:

- Classification model: High precision and recall for known fraud patterns
- Autoencoder model: Effective at detecting novel and unusual transaction patterns
- Ensemble approach: Combines strengths of both models for robust detection

Detailed performance metrics and visualizations are generated in the `/results` directory.

## 🔄 Model Monitoring

The system includes comprehensive monitoring capabilities:

- Data drift detection between training and production data
- Model performance tracking over time
- Automated reporting with recommendations for model updates

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Kaggle for providing the financial transactions dataset
- The Apache Spark and TensorFlow communities for their excellent tools
- Contributors to the various open-source libraries used in this project
