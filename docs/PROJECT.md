# Anomaly-Based Fraud Detection in Banking Transactions Utilising Spark and Deep Learning Techniques

## Project Overview

The goal of this thesis project is to develop an **anomaly-based fraud detection system** for banking transactions. We aim to leverage big-data processing and deep learning to identify unusual transaction patterns that indicate fraud. The key technologies and tools include:

* **PySpark** : Distributed data processing for handling large-scale transaction data efficiently.
* **TensorFlow or PyTorch** : Deep learning frameworks for designing and training neural network models (e.g. autoencoders or classification networks).
* **MLflow** : Experiment tracking and model management to record parameters, metrics, and artifacts.
* **Jupyter Notebooks** : Interactive environment for exploratory data analysis (EDA), visualization, and documenting experimentation.

Together, these technologies will allow for scalable data ingestion and preprocessing, sophisticated model building, and rigorous experiment tracking in a streamlined workflow.

## Dataset Description

* **Source:** *Kaggle Financial Transactions Dataset for Fraud Detection* (synthetic data).
* **Contents:** Approximately **5 million** simulated banking/mobile money transactions. The dataset is generated to mimic real-world financial operations with diverse user behavior.
* **Fields:** Each record includes transaction attributes such as:
  * **Transaction Details:** ID, timestamp/date, amount, sender account, receiver account, merchant or category.
  * **Behavioral Features:** Features derived from user behavior, e.g. average transaction value, frequency of transactions in last 24h, velocity features.
  * **Metadata:** Device type, location (country, city), payment method, merchant category code.
  * **Fraud Labels:** A binary label (fraudulent or legitimate). In some versions, a multi-class fraud type may be included for granular anomaly categories.
* **Use Cases:**
  * **Fraud Detection (Binary/Multiclass):** Classify each transaction as fraudulent or not (or into fraud types), using supervised or semi-supervised models.
  * **Time-series Anomaly Detection:** Monitor transaction streams for unusual spikes or outliers over time, potentially using unsupervised models (e.g. autoencoders or anomaly scoring) on temporal features.

## Project Structure

The project is organized into a clear directory hierarchy to separate data, code, and documentation:

```
/data
  /raw           # Original raw data files (e.g. CSV from Kaggle)
  /processed     # Cleaned and transformed data (e.g. Parquet/CSV for modeling)
  
/notebooks       # Jupyter notebooks for EDA, prototyping, and reporting

/src             # Source code for data pipelines and models
  /spark_jobs      # PySpark ETL scripts for data loading and preprocessing
  /models          # Model definitions, architectures, and training scripts
  /utils           # Utility modules (e.g. data loaders, feature engineering helpers)

/results        # Output files such as evaluation metrics, figures, logs

/docs           # Project documentation, design notes, and report drafts

README.md       # Project overview, setup instructions, and usage guide
```

* Each directory contains relevant files: raw data ingested into `/data/raw`, processing code in `/src/spark_jobs`, modeling code in `/src/models`, etc.
* The **README.md** at the root provides an overview of the project, setup instructions, and how to reproduce experiments.

## Partial Implementations

### PySpark Script (Loading & Preprocessing)

This placeholder script demonstrates how to load and preprocess data using PySpark. It initializes a Spark session, reads raw CSV data, and includes spots for data cleaning and feature engineering.

```python
# src/spark_jobs/load_data.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

def load_data(file_path):
    spark = SparkSession.builder.appName("FraudDetectionPreprocessing").getOrCreate()
    # Read raw transaction data (assuming CSV with header and inferSchema)
    df = spark.read.csv(file_path, header=True, inferSchema=True)
    return df

def preprocess_data(df):
    # Placeholder: data cleaning and feature engineering steps
    # Example: drop rows with null values in critical columns
    df = df.dropna(subset=["amount", "timestamp", "sender"])
    # Convert timestamp column to proper type (if needed)
    # df = df.withColumn("timestamp", col("timestamp").cast("timestamp"))
    # Feature engineering (e.g., create amount_bins, time_of_day features)
    # df = df.withColumn("amount_log", log(col("amount") + 1))
    # Add more preprocessing as required
    return df

if __name__ == "__main__":
    raw_path = "data/raw/transactions.csv"
    processed_path = "data/processed/transactions.parquet"
    df_raw = load_data(raw_path)
    df_processed = preprocess_data(df_raw)
    # Save the processed data for modeling (e.g., as Parquet for efficiency)
    df_processed.write.mode("overwrite").parquet(processed_path)
```

### Deep Learning Model (TensorFlow)

We've implemented advanced neural network architectures for fraud detection using TensorFlow/Keras. The system includes both classification models for supervised learning and autoencoder models for unsupervised anomaly detection.

#### Classification Model

```python
# src/models/train_model.py (simplified version)
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def create_classification_model(input_dim, hidden_layers=[256, 128, 64, 32], dropout_rate=0.5, l2_regularization=0.001):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(input_dim,)))
    
    # Add hidden layers with L2 regularization and dropout
    for units in hidden_layers:
        model.add(layers.Dense(units, activation='relu', 
                              kernel_regularizer=regularizers.l2(l2_regularization)))
        model.add(layers.Dropout(dropout_rate))
    
    # Output layer for binary classification
    model.add(layers.Dense(1, activation='sigmoid'))
    
    # Compile with custom loss function for imbalanced data
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=asymmetric_focal_loss(gamma=4.0, alpha=0.3),
        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Recall(), 
                 tf.keras.metrics.Precision()]
    )
    return model
```

#### Autoencoder Model

```python
# Autoencoder for unsupervised anomaly detection
def create_autoencoder_model(input_dim, encoding_dim=24, hidden_layers=[128, 64], l2_regularization=0.001):
    # Encoder
    input_layer = layers.Input(shape=(input_dim,))
    encoded = input_layer
    
    for units in hidden_layers:
        encoded = layers.Dense(units, activation='relu',
                              kernel_regularizer=regularizers.l2(l2_regularization))(encoded)
    
    # Bottleneck layer
    bottleneck = layers.Dense(encoding_dim, activation='relu',
                            kernel_regularizer=regularizers.l2(l2_regularization))(encoded)
    
    # Decoder (mirror of encoder)
    decoded = bottleneck
    for units in reversed(hidden_layers):
        decoded = layers.Dense(units, activation='relu',
                              kernel_regularizer=regularizers.l2(l2_regularization))(decoded)
    
    # Output reconstruction
    output_layer = layers.Dense(input_dim, activation='linear')(decoded)
    
    # Create model
    model = models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mse')
    
    return model
```

### MLflow Configuration (Placeholder)

An example MLflow setup for experiment tracking. We define an experiment and demonstrate logging of parameters and metrics. In practice, this would be integrated into the training script.

```python
# mlflow_example.py
import mlflow

# Set up MLflow experiment (name and tracking URI can be customized)
mlflow.set_experiment("Fraud_Detection_Experiment")

with mlflow.start_run():
    # Example parameter logging
    mlflow.log_param("model_type", "neural_network")
    mlflow.log_param("epochs", 10)
    # Placeholder for model training and evaluation
    # After training:
    mlflow.log_metric("validation_accuracy", 0.95)
    mlflow.log_metric("validation_auc", 0.97)
    # Log the model artifact (if applicable)
    # mlflow.keras.log_model(model, "model")
    # End of run (automatically done at context exit)
```

Alternatively, an `mlflow_config.yaml` could be used for configuration:

```yaml
# mlflow_config.yaml
experiment:
  name: "Fraud_Detection_Experiment"
tracking_uri: "http://localhost:5000"  # MLflow tracking server URI
```

### Jupyter Notebook Scaffold

The main analysis notebook outlines the workflow with sections for EDA, preprocessing, modeling, and evaluation. Below is a template structure for `notebooks/Fraud_Detection_Analysis.ipynb`:

```markdown
# Fraud Detection Analysis

## 1. Exploratory Data Analysis (EDA)
* Load a sample of the transaction data
* Compute summary statistics (means, transaction count, fraud rate)
* Visualize distributions of amount, transaction times, and fraud frequency

## 2. Data Preprocessing
* Handle missing values and outliers
* Encode categorical features (e.g., one-hot for merchant category)
* Normalize or scale numeric features (amount, time-based features)
* Split data into train/validation/test sets

## 3. Modeling
* Build a PySpark ML pipeline for feature assembly (if needed)
* Instantiate and train the deep learning model (e.g. with TensorFlow/Keras)
* Use MLflow to log training runs and parameters

## 4. Evaluation
* Calculate performance metrics: Accuracy, Precision, Recall, F1-score, ROC AUC
* Plot confusion matrix and ROC curve
* Analyze detected anomalies and false positives
```

## Research Report Outline

* **Introduction:** Context and motivation for fraud detection in banking; objectives of the thesis project.
* **Literature Review:** Survey of existing fraud detection techniques, anomaly detection methods, and use of big data frameworks in finance.
* **Methodology:** Detailed description of the dataset, Spark-based preprocessing pipeline, feature engineering, and the deep learning modeling approach.
* **Results:** Presentation of experimental findings, model performance metrics, comparison with baseline methods, and visualizations (e.g., ROC curves).
* **Discussion:** Interpretation of results, strengths and limitations of the proposed approach, handling of imbalanced classes, and real-world applicability.
* **Conclusion:** Summary of key contributions, implications for banking fraud prevention, and suggestions for future research (e.g., incremental learning, deployment considerations).

All documentation, code comments, and descriptions are written in English to ensure clarity and consistency throughout the project.
