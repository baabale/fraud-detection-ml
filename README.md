# Anomaly-Based Fraud Detection in Banking Transactions

This project implements an anomaly-based fraud detection system for banking transactions using Spark and deep learning techniques.

## Project Overview

The goal of this project is to develop a scalable system that can identify unusual transaction patterns indicating fraud in banking data. The system leverages:

- **PySpark**: For distributed data processing of large-scale transaction data
- **Deep Learning** (TensorFlow/PyTorch): For building sophisticated neural network models
- **MLflow**: For experiment tracking and model management
- **Jupyter Notebooks**: For exploratory data analysis and visualization

## Dataset

The project uses the Kaggle Financial Transactions Dataset for Fraud Detection, which contains approximately 5 million simulated banking transactions with various attributes including:
- Transaction details (ID, timestamp, amount, sender/receiver accounts)
- Behavioral features
- Metadata (device type, location, payment method)
- Fraud labels

## Project Structure

```
/data
  /raw           # Original raw data files (e.g., CSV from Kaggle)
  /processed     # Cleaned and transformed data

/notebooks       # Jupyter notebooks for EDA, prototyping, and reporting

/src             # Source code for data pipelines and models
  /spark_jobs    # PySpark ETL scripts for data loading and preprocessing
  /models        # Model definitions, architectures, and training scripts
  /utils         # Utility modules (data loaders, feature engineering helpers)

/results         # Output files (evaluation metrics, figures, logs)

/docs            # Project documentation and design notes
```

## Setup Instructions

### Prerequisites

- Python 3.8+
- Apache Spark 3.x
- TensorFlow 2.x or PyTorch 1.x
- MLflow
- Jupyter Notebook/Lab

### Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd thesis
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

4. Download the dataset from Kaggle and place it in the `/data/raw` directory.

## Usage

### Data Preprocessing

Run the PySpark preprocessing script:
```
python src/spark_jobs/load_data.py
```

### Exploratory Data Analysis

Open and run the Jupyter notebooks in the `/notebooks` directory:
```
jupyter notebook notebooks/
```

### Model Training

Train the fraud detection model:
```
python src/models/train_model.py
```

### Experiment Tracking

View MLflow experiments:
```
mlflow ui
```

## License

[Specify license information]
