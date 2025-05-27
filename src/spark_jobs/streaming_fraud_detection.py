"""
Real-time fraud detection using Spark Structured Streaming.
This script processes transaction streams and applies the trained models for fraud detection.
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, struct, to_json, from_json
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, 
    TimestampType, BooleanType, ArrayType
)
import tensorflow as tf
import joblib
from datetime import datetime

# Configure TensorFlow to use GPU if available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth for all GPUs to prevent TensorFlow from allocating all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("\n" + "="*70)
        print(f"🚀 GPU ACCELERATION ENABLED: Using {len(gpus)} GPU(s) for streaming inference")
        print("="*70 + "\n")
        
        # Configure multi-GPU strategy by default if multiple GPUs are available
        if len(gpus) > 1:
            try:
                print(f"\n" + "="*70)
                print(f"🚀 MULTI-GPU STREAMING ENABLED: Distributing across {len(gpus)} GPUs")
                print("="*70 + "\n")
                
                # Create a MirroredStrategy for multi-GPU inference
                strategy = tf.distribute.MirroredStrategy()
                print(f"Number of devices in sync: {strategy.num_replicas_in_sync}")
            except Exception as e:
                print(f"Error configuring multi-GPU strategy: {e}")
                strategy = None
        else:
            strategy = None
    except RuntimeError as e:
        print(f"Error configuring GPUs: {e}")
        strategy = None
else:
    print("\n" + "="*70)
    print("⚠️ NO GPU DETECTED: Using CPU for streaming inference (this will be slower)")
    print("="*70 + "\n")
    strategy = None

# Add the project root to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the configuration manager
from src.utils.config_manager import config

def create_spark_session(app_name=None, gpu_available=None, multi_gpu=True):
    """
    Create and return a Spark session configured for streaming.
    
    Args:
        app_name (str, optional): Name of the Spark application. If None, uses the value from config.
        gpu_available (bool, optional): Whether GPU is available. If None, will be auto-detected.
        multi_gpu (bool): Whether to configure for multiple GPUs. Default is True.
        
    Returns:
        SparkSession: Configured Spark session
    """
    # Get Spark configuration from config
    spark_config = config.get_spark_config()
    app_name = app_name or spark_config.get('app_name', 'FraudDetectionStreaming')
    driver_memory = spark_config.get('driver_memory', '4g')
    executor_memory = spark_config.get('executor_memory', '4g')
    
    # Auto-detect GPU availability if not explicitly provided
    if gpu_available is None:
        gpu_available = False
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            gpu_available = len(gpus) > 0
        except ImportError:
            pass
    
    # Start building the Spark session
    builder = SparkSession.builder.appName(app_name)
    
    # Configure basic Spark settings
    builder = builder.config("spark.streaming.stopGracefullyOnShutdown", "true")
    builder = builder.config("spark.driver.memory", driver_memory)
    builder = builder.config("spark.executor.memory", executor_memory)
    
    # Configure GPU acceleration if available
    if gpu_available:
        print("\n" + "="*70)
        print(f"🚀 CONFIGURING SPARK STREAMING FOR GPU ACCELERATION")
        print("="*70 + "\n")
        
        # Enable GPU acceleration for Spark
        builder = builder.config("spark.rapids.sql.enabled", "true")
        builder = builder.config("spark.rapids.memory.pinnedPool.size", "2G")
        builder = builder.config("spark.sql.execution.arrow.pyspark.enabled", "true")
        builder = builder.config("spark.sql.execution.arrow.maxRecordsPerBatch", "500000")
        
        # Configure for multi-GPU if requested and available
        if multi_gpu:
            try:
                import tensorflow as tf
                gpus = tf.config.list_physical_devices('GPU')
                if len(gpus) > 1:
                    print(f"Configuring Spark for {len(gpus)} GPUs...")
                    # Set concurrent tasks based on GPU count
                    builder = builder.config("spark.rapids.sql.concurrentGpuTasks", str(len(gpus)))
                    builder = builder.config("spark.rapids.sql.explain", "ALL")
                    builder = builder.config("spark.rapids.memory.gpu.pooling.enabled", "true")
                    builder = builder.config("spark.rapids.sql.multiThreadedRead.numThreads", str(len(gpus) * 2))
                else:
                    print("Only one GPU detected. Using single-GPU configuration.")
                    builder = builder.config("spark.rapids.sql.concurrentGpuTasks", "2")
                    builder = builder.config("spark.rapids.sql.explain", "ALL")
            except ImportError:
                # Fall back to default GPU settings if TensorFlow is not available
                builder = builder.config("spark.rapids.sql.concurrentGpuTasks", "2")
                builder = builder.config("spark.rapids.sql.explain", "ALL")
        else:
            # Single GPU configuration
            builder = builder.config("spark.rapids.sql.concurrentGpuTasks", "2")
            builder = builder.config("spark.rapids.sql.explain", "ALL")
    else:
        print("Using CPU-only configuration for Spark")
    
    # Create and return the session
    return builder.getOrCreate()

def load_models(model_dir):
    """
    Load trained models and preprocessors.
    
    Args:
        model_dir (str): Directory containing the models
        
    Returns:
        tuple: Loaded models and preprocessors
    """
    print(f"Loading models from {model_dir}")
    
    # Load scaler
    scaler_path = os.path.join(model_dir, 'scaler.joblib')
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print(f"Loaded scaler from {scaler_path}")
    else:
        scaler = None
        print(f"Warning: Scaler not found at {scaler_path}")
    
    # Load feature names
    feature_names_path = os.path.join(model_dir, 'feature_names.json')
    if os.path.exists(feature_names_path):
        with open(feature_names_path, 'r') as f:
            feature_names = json.load(f)
        print(f"Loaded {len(feature_names)} feature names")
    else:
        feature_names = None
        print(f"Warning: Feature names not found at {feature_names_path}")
    
    # Load autoencoder threshold
    threshold_path = os.path.join(model_dir, 'autoencoder_threshold.json')
    if os.path.exists(threshold_path):
        with open(threshold_path, 'r') as f:
            threshold_data = json.load(f)
            autoencoder_threshold = threshold_data.get('threshold', 0.1)
        print(f"Loaded autoencoder threshold: {autoencoder_threshold}")
    else:
        autoencoder_threshold = 0.1
        print(f"Warning: Autoencoder threshold not found, using default: {autoencoder_threshold}")
    
    # Load classification model with multi-GPU support if available
    classification_model_path = os.path.join(model_dir, 'classification_model.h5')
    if os.path.exists(classification_model_path):
        # Use strategy scope if multi-GPU is enabled
        if 'strategy' in globals() and strategy is not None:
            print(f"Loading classification model with multi-GPU strategy")
            with strategy.scope():
                classification_model = tf.keras.models.load_model(classification_model_path)
        else:
            classification_model = tf.keras.models.load_model(classification_model_path)
        print(f"Loaded classification model from {classification_model_path}")
    else:
        classification_model = None
        print(f"Warning: Classification model not found at {classification_model_path}")
    
    # Load autoencoder model with multi-GPU support if available
    autoencoder_model_path = os.path.join(model_dir, 'autoencoder_model.h5')
    if os.path.exists(autoencoder_model_path):
        # Use strategy scope if multi-GPU is enabled
        if 'strategy' in globals() and strategy is not None:
            print(f"Loading autoencoder model with multi-GPU strategy")
            with strategy.scope():
                autoencoder_model = tf.keras.models.load_model(autoencoder_model_path)
        else:
            autoencoder_model = tf.keras.models.load_model(autoencoder_model_path)
        print(f"Loaded autoencoder model from {autoencoder_model_path}")
    else:
        autoencoder_model = None
        print(f"Warning: Autoencoder model not found at {autoencoder_model_path}")
    
    return classification_model, autoencoder_model, scaler, feature_names, autoencoder_threshold

def preprocess_data(df, feature_names, scaler):
    """
    Preprocess a DataFrame for model inference.
    
    Args:
        df (DataFrame): Input DataFrame
        feature_names (list): List of feature names
        scaler: Fitted scaler for feature normalization
        
    Returns:
        DataFrame: Preprocessed DataFrame
    """
    # Select only the required features
    if feature_names is not None:
        # Ensure all required features are present
        for feature in feature_names:
            if feature not in df.columns:
                df = df.withColumn(feature, col(feature).cast(DoubleType()))
        
        # Select and order features
        df = df.select(*feature_names)
    
    # Convert to Pandas DataFrame for scaling
    pdf = df.toPandas()
    
    # Scale features if scaler is available
    if scaler is not None:
        X = scaler.transform(pdf.values)
        pdf = pd.DataFrame(X, columns=pdf.columns)
    
    return pdf

def compute_anomaly_scores(model, X):
    """
    Compute anomaly scores using the autoencoder reconstruction error.
    
    Args:
        model: Trained autoencoder model
        X: Input data
        
    Returns:
        array: Anomaly scores
    """
    # Get reconstructions
    X_pred = model.predict(X.values)
    
    # Compute mean squared error for each sample
    mse = np.mean(np.square(X.values - X_pred), axis=1)
    
    return mse

def process_batch(batch_df, epoch_id, classification_model, autoencoder_model, 
                 scaler, feature_names, autoencoder_threshold):
    """
    Process a batch of transactions for fraud detection.
    
    Args:
        batch_df: Batch DataFrame
        epoch_id: Epoch ID
        classification_model: Trained classification model
        autoencoder_model: Trained autoencoder model
        scaler: Fitted scaler for feature normalization
        feature_names: List of feature names
        autoencoder_threshold: Threshold for autoencoder anomaly detection
        
    Returns:
        DataFrame: Processed DataFrame with fraud predictions
    """
    # Skip empty batches
    if batch_df.isEmpty():
        return batch_df
    
    # Log batch information
    print(f"-------------------------------------------")
    print(f"Batch: {epoch_id} with {batch_df.count()} transactions")
    print(f"-------------------------------------------")
    
    # Check if multi-GPU strategy is available
    using_multi_gpu = 'strategy' in globals() and strategy is not None
    if using_multi_gpu:
        print(f"Using multi-GPU strategy for batch processing")
    
    # Preprocess data
    preprocessed_df = preprocess_data(batch_df, feature_names, scaler)
    
    # Make predictions
    results = []
    
    # Classification predictions with multi-GPU optimization
    if classification_model is not None:
        start_time = datetime.now()
        
        if using_multi_gpu:
            # Create TensorFlow dataset with prefetching for multi-GPU inference
            X_tensor = tf.convert_to_tensor(preprocessed_df.values, dtype=tf.float32)
            batch_size = min(len(preprocessed_df), 1024)  # Use appropriate batch size
            tf_dataset = tf.data.Dataset.from_tensor_slices(X_tensor)
            tf_dataset = tf_dataset.batch(batch_size)
            tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)
            
            # Run inference in batches
            classification_probs_list = []
            for batch in tf_dataset:
                batch_probs = classification_model.predict(batch, verbose=0)
                classification_probs_list.append(batch_probs)
            
            # Combine results
            classification_probs = np.vstack(classification_probs_list).flatten()
        else:
            # Standard inference
            classification_probs = classification_model.predict(preprocessed_df.values).flatten()
        
        classification_preds = (classification_probs >= 0.5).astype(int)
        inference_time = (datetime.now() - start_time).total_seconds()
        print(f"Classification inference completed in {inference_time:.3f} seconds")
    else:
        classification_probs = np.zeros(len(preprocessed_df))
        classification_preds = np.zeros(len(preprocessed_df))
    
    # Autoencoder predictions with multi-GPU optimization
    if autoencoder_model is not None:
        start_time = datetime.now()
        
        if using_multi_gpu:
            # Create TensorFlow dataset with prefetching for multi-GPU inference
            X_tensor = tf.convert_to_tensor(preprocessed_df.values, dtype=tf.float32)
            batch_size = min(len(preprocessed_df), 1024)  # Use appropriate batch size
            tf_dataset = tf.data.Dataset.from_tensor_slices(X_tensor)
            tf_dataset = tf_dataset.batch(batch_size)
            tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)
            
            # Run inference in batches
            reconstructions_list = []
            for batch in tf_dataset:
                batch_recon = autoencoder_model.predict(batch, verbose=0)
                reconstructions_list.append(batch_recon)
            
            # Combine results and calculate anomaly scores
            reconstructions = np.vstack(reconstructions_list)
            anomaly_scores = np.mean(np.square(preprocessed_df.values - reconstructions), axis=1)
        else:
            # Standard inference
            anomaly_scores = compute_anomaly_scores(autoencoder_model, preprocessed_df)
        
        autoencoder_preds = (anomaly_scores >= autoencoder_threshold).astype(int)
        inference_time = (datetime.now() - start_time).total_seconds()
        print(f"Autoencoder inference completed in {inference_time:.3f} seconds")
    else:
        anomaly_scores = np.zeros(len(preprocessed_df))
        autoencoder_preds = np.zeros(len(preprocessed_df))
    
    # Combine results
    for i in range(len(preprocessed_df)):
        # Get original transaction data
        transaction = batch_df.collect()[i]
        
        # Create result with predictions
        result = {
            'transaction_id': transaction.get('transaction_id', f"tx_{i}_{epoch_id}"),
            'timestamp': transaction.get('timestamp', datetime.now().isoformat()),
            'fraud_probability': float(classification_probs[i]),
            'anomaly_score': float(anomaly_scores[i]),
            'is_fraud_classification': bool(classification_preds[i]),
            'is_fraud_autoencoder': bool(autoencoder_preds[i]),
            'is_fraud': bool(classification_preds[i] or autoencoder_preds[i])
        }
        
        # Add original transaction data
        for field in batch_df.schema.fieldNames():
            if field not in result:
                result[field] = transaction[field]
        
        results.append(result)
    
    # Print summary of fraud detections
    fraud_count = sum(1 for r in results if r['is_fraud'])
    if fraud_count > 0:
        print(f"\n!!! ALERT: {fraud_count} fraudulent transactions detected in this batch !!!\n")
    else:
        print("\nNo fraudulent transactions detected in this batch\n")
    
    # Convert back to Spark DataFrame
    spark = SparkSession.builder.getOrCreate()
    return spark.createDataFrame(result_df)

def define_schema():
    """
    Define the schema for incoming transaction data.
    
    Returns:
        StructType: Schema for transaction data
    """
    return StructType([
        StructField("transaction_id", StringType(), True),
        StructField("timestamp", TimestampType(), True),
        StructField("amount", DoubleType(), True),
        StructField("sender_account", StringType(), True),
        StructField("receiver_account", StringType(), True),
        StructField("merchant_category", StringType(), True),
        # Add more fields as needed
    ])

def main():
    """
    Main function to run the streaming fraud detection pipeline.
    """
    parser = argparse.ArgumentParser(description='Run streaming fraud detection')
    parser.add_argument('--model-dir', type=str,
                        help='Directory containing the trained models')
    parser.add_argument('--input-format', type=str, choices=['kafka', 'socket', 'file'],
                        help='Input stream format')
    parser.add_argument('--input-path', type=str,
                        help='Input path/topic/socket (depends on format)')
    parser.add_argument('--output-format', type=str, choices=['console', 'kafka', 'file'],
                        help='Output format')
    parser.add_argument('--output-path', type=str,
                        help='Output path/topic (depends on format)')
    parser.add_argument('--checkpoint-dir', type=str,
                        help='Directory for streaming checkpoints')
    # GPU configuration
    parser.add_argument('--disable-gpu', action='store_true',
                        help='Disable GPU usage even if available')
    parser.add_argument('--single-gpu', action='store_true',
                        help='Use only a single GPU even if multiple are available')
    parser.add_argument('--memory-growth', action='store_true',
                        help='Enable memory growth for GPUs to prevent TensorFlow from allocating all memory')
    args = parser.parse_args()
    
    # Use configuration values if arguments are not provided
    model_dir = args.model_dir or config.get_model_path()
    input_format = args.input_format or config.get('streaming.input_format', 'socket')
    input_path = args.input_path or config.get('streaming.input_path', 'localhost:9999')
    output_format = args.output_format or config.get('streaming.output_format', 'console')
    output_path = args.output_path or config.get('streaming.output_path')
    checkpoint_dir = args.checkpoint_dir or config.get_absolute_path(config.get('streaming.checkpoint_dir', 'checkpoints'))
    
    # Handle GPU configuration based on command-line arguments
    gpu_available = None  # Auto-detect by default
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        
        if args.disable_gpu:
            print("\n" + "="*70)
            print("⚠️ GPU DISABLED: Using CPU for streaming inference as requested")
            print("="*70 + "\n")
            tf.config.set_visible_devices([], 'GPU')
            gpu_available = False
        elif gpus:
            if args.single_gpu and len(gpus) > 1:
                # Use only the first GPU
                tf.config.set_visible_devices(gpus[0], 'GPU')
                print(f"\n" + "="*70)
                print(f"🚀 SINGLE GPU MODE: Using only one GPU ({gpus[0]}) for streaming inference")
                print("="*70 + "\n")
            
            if args.memory_growth:
                # Enable memory growth for all GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("\n" + "="*70)
                print(f"🚀 MEMORY GROWTH ENABLED: GPUs will allocate memory as needed")
                print("="*70 + "\n")
        else:
            print("\n" + "="*70)
            print("⚠️ NO GPU DETECTED: Using CPU for streaming inference")
            print("="*70 + "\n")
    except ImportError:
        print("TensorFlow not available, using CPU for streaming inference")
        gpu_available = False
    
    # Create Spark session with appropriate GPU configuration
    # Use multi-GPU by default unless single-GPU is specified
    use_multi_gpu = not args.single_gpu
    spark = create_spark_session(gpu_available=gpu_available, multi_gpu=use_multi_gpu)
    
    # Load models
    classification_model, autoencoder_model, scaler, feature_names, autoencoder_threshold = load_models(model_dir)
    
    # Define schema
    schema = define_schema()
    
    # Create streaming DataFrame based on input format
    if input_format == 'kafka':
        stream_df = (spark.readStream
                    .format("kafka")
                    .option("kafka.bootstrap.servers", input_path)
                    .option("subscribe", "transactions")
                    .option("startingOffsets", "latest")
                    .load()
                    .selectExpr("CAST(value AS STRING) as json")
                    .select(from_json("json", schema).alias("data"))
                    .select("data.*"))
    elif input_format == 'socket':
        stream_df = (spark.readStream
                    .format("socket")
                    .option("host", input_path.split(':')[0])
                    .option("port", input_path.split(':')[1])
                    .load()
                    .selectExpr("CAST(value AS STRING) as json")
                    .select(from_json("json", schema).alias("data"))
                    .select("data.*"))
    elif input_format == 'file':
        stream_df = (spark.readStream
                    .format("json")
                    .schema(schema)
                    .option("path", input_path)
                    .option("maxFilesPerTrigger", 1)
                    .load())
    
    # Process each batch with the fraud detection models
    query = (stream_df.writeStream
            .foreachBatch(lambda batch_df, epoch_id: process_batch(
                batch_df, epoch_id, classification_model, autoencoder_model, 
                scaler, feature_names, autoencoder_threshold
            ))
            .outputMode("append")
            .trigger(processingTime="1 second"))
    
    # Configure output
    if output_format == 'console':
        query = (query.format("console")
                .option("truncate", False))
    elif output_format == 'kafka':
        query = (query.format("kafka")
                .option("kafka.bootstrap.servers", output_path)
                .option("topic", "fraud_predictions"))
    elif output_format == 'file':
        if output_path is None:
            output_path = config.get_absolute_path("results/streaming_output")
        query = (query.format("json")
                .option("path", output_path)
                .option("checkpointLocation", checkpoint_dir))
    
    # Start the streaming query
    streaming_query = query.start()
    
    # Wait for the query to terminate
    try:
        streaming_query.awaitTermination()
    except KeyboardInterrupt:
        print("Stopping streaming query...")
        streaming_query.stop()
        print("Query stopped.")

if __name__ == "__main__":
    main()
