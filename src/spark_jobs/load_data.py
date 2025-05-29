"""
PySpark script for loading and preprocessing banking transaction data for fraud detection.
"""
import os
import sys
import argparse
import logging
import warnings
import time
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, log, hour, dayofweek, month, year, datediff, current_date, \
    when, lit, concat
from pyspark.sql.types import TimestampType

# Filter PySpark warnings to avoid excessive logging
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*jdk.incubator.*')
warnings.filterwarnings('ignore', message='.*Your hostname.*')
warnings.filterwarnings('ignore', message='.*Set SPARK_LOCAL_IP.*')
warnings.filterwarnings('ignore', message='.*Unable to load native-hadoop.*')

# Try to import TensorFlow for GPU detection
try:
    import tensorflow as tf
    import logging
    
    # Get a logger
    gpu_logger = logging.getLogger('fraud_detection')
    
    # Configure TensorFlow to use GPU if available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth for all GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            gpu_message = f"\n{'='*70}\n🚀 GPU ACCELERATION ENABLED: Using {len(gpus)} GPU(s) for data processing\n{'='*70}\n"
            gpu_logger.info(gpu_message)
            
            # Test GPU performance
            gpu_logger.info("Testing GPU performance for data processing...")
            x = tf.random.normal([1000, 1000])
            start = time.time()
            tf.matmul(x, x)
            gpu_logger.info(f"Matrix multiplication took: {time.time() - start:.6f} seconds")
            gpu_logger.info(f"TensorFlow version: {tf.__version__}")
        except RuntimeError as e:
            gpu_logger.error(f"Error configuring GPUs: {e}")
    else:
        gpu_message = f"\n{'='*70}\n⚠️ NO GPU DETECTED: Using CPU for data processing (this will be slower)\n{'='*70}\n"
        gpu_logger.warning(gpu_message)
except ImportError:
    # If we can't import tensorflow, we need to set up a basic logger
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('fraud_detection')
    logger.warning("TensorFlow not available. GPU acceleration not configured for data processing.")


def create_spark_session(app_name="FraudDetectionPreprocessing", gpu_available=None, multi_gpu=False, gpu_memory="2G"):
    """
    Create and return a Spark session.
    
    Args:
        app_name (str): Name of the Spark application
        gpu_available (bool, optional): Whether GPU is available. If None, will be auto-detected.
        multi_gpu (bool): Whether to configure for multiple GPUs
        gpu_memory (str): Amount of GPU memory to allocate (e.g., "2G")
        
    Returns:
        SparkSession: Configured Spark session
    """
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
    
    # Configure memory
    builder = builder.config("spark.driver.memory", "4g") \
                   .config("spark.executor.memory", "4g")
    
    # Configure GPU acceleration if available
    if gpu_available:
        # Get a logger
        logger = logging.getLogger('fraud_detection')
        
        gpu_message = f"\n{'='*70}\n🚀 CONFIGURING SPARK FOR GPU ACCELERATION\n{'='*70}\n"
        logger.info(gpu_message)
        
        # Enable GPU acceleration for Spark
        builder = builder.config("spark.rapids.sql.enabled", "true") \
                       .config("spark.rapids.memory.pinnedPool.size", gpu_memory) \
                       .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
                       .config("spark.sql.execution.arrow.maxRecordsPerBatch", "500000")
        
        # Configure for multi-GPU if requested and available
        if multi_gpu:
            try:
                import tensorflow as tf
                gpus = tf.config.list_physical_devices('GPU')
                if len(gpus) > 1:
                    logger.info(f"Configuring Spark for {len(gpus)} GPUs...")
                    # Set concurrent tasks based on GPU count
                    builder = builder.config("spark.rapids.sql.concurrentGpuTasks", str(len(gpus))) \
                               .config("spark.rapids.sql.explain", "ALL") \
                               .config("spark.rapids.memory.gpu.pooling.enabled", "true") \
                               .config("spark.rapids.sql.multiThreadedRead.numThreads", str(len(gpus) * 2))
                else:
                    print("Only one GPU detected. Using single-GPU configuration.")
                    builder = builder.config("spark.rapids.sql.concurrentGpuTasks", "2") \
                               .config("spark.rapids.sql.explain", "ALL")
            except ImportError:
                # Fall back to default GPU settings if TensorFlow is not available
                builder = builder.config("spark.rapids.sql.concurrentGpuTasks", "2") \
                           .config("spark.rapids.sql.explain", "ALL")
        else:
            # Single GPU configuration
            builder = builder.config("spark.rapids.sql.concurrentGpuTasks", "2") \
                           .config("spark.rapids.sql.explain", "ALL")
    else:
        print("Using CPU-only configuration for Spark")
    
    # Create and return the session
    return builder.getOrCreate()

def load_data(spark, file_path):
    """
    Load transaction data from CSV file.
    
    Args:
        spark (SparkSession): Active Spark session
        file_path (str): Path to the raw data file
        
    Returns:
        DataFrame: Spark DataFrame containing the raw transaction data
    """
    import os
    from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, TimestampType
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} does not exist!")
        print("Creating synthetic data for demonstration purposes...")
        
        # Create a schema for synthetic data
        schema = StructType([
            StructField("transaction_id", StringType(), False),
            StructField("timestamp", TimestampType(), False),
            StructField("amount", DoubleType(), False),
            StructField("sender", StringType(), False),
            StructField("receiver", StringType(), False),
            StructField("transaction_type", StringType(), False),
            StructField("is_fraud", IntegerType(), False)
        ])
        
        # Create an empty dataframe with the schema
        df = spark.createDataFrame([], schema)
        
        # Generate synthetic data using Spark SQL
        df = spark.sql("""
        SELECT 
            concat('TX', cast(id as string)) as transaction_id,
            current_timestamp() - (rand() * 86400 * 30) as timestamp,
            rand() * 10000 as amount,
            concat('SENDER', cast(floor(rand() * 1000) as string)) as sender,
            concat('RECEIVER', cast(floor(rand() * 1000) as string)) as receiver,
            case floor(rand() * 4)
                when 0 then 'PAYMENT'
                when 1 then 'TRANSFER'
                when 2 then 'WITHDRAWAL'
                else 'DEPOSIT'
            end as transaction_type,
            case when rand() < 0.05 then 1 else 0 end as is_fraud
        FROM range(1000)
        """)
        
        print(f"Created synthetic dataset with {df.count()} transactions")
        return df
    
    # Read raw transaction data (assuming CSV with header and inferSchema)
    try:
        df = spark.read.csv(file_path, header=True, inferSchema=True)
        print(f"Loaded {df.count()} transactions from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        print("Creating synthetic data for demonstration purposes...")
        
        # Generate synthetic data (same as above)
        df = spark.sql("""
        SELECT 
            concat('TX', cast(id as string)) as transaction_id,
            current_timestamp() - (rand() * 86400 * 30) as timestamp,
            rand() * 10000 as amount,
            concat('SENDER', cast(floor(rand() * 1000) as string)) as sender,
            concat('RECEIVER', cast(floor(rand() * 1000) as string)) as receiver,
            case floor(rand() * 4)
                when 0 then 'PAYMENT'
                when 1 then 'TRANSFER'
                when 2 then 'WITHDRAWAL'
                else 'DEPOSIT'
            end as transaction_type,
            case when rand() < 0.05 then 1 else 0 end as is_fraud
        FROM range(1000)
        """)
        
        print(f"Created synthetic dataset with {df.count()} transactions")
        return df

def preprocess_data(df):
    """
    Preprocess transaction data by cleaning, transforming, and engineering features.
    
    Args:
        df (DataFrame): Raw transaction DataFrame
        
    Returns:
        DataFrame: Processed DataFrame ready for modeling
    """
    # Get a logger
    logger = logging.getLogger('fraud_detection')
    
    logger.info("Starting data preprocessing...")
    
    # List available columns for debugging
    logger.info(f"Available columns: {df.columns}")
    
    # Drop rows with null values in critical columns
    df = df.dropna(subset=["amount", "timestamp", "sender_account"])
    
    # Convert timestamp column to proper type (if needed)
    if "timestamp" in df.columns:
        df = df.withColumn("timestamp", col("timestamp").cast(TimestampType()))
    
    # Feature engineering
    logger.info("Performing feature engineering...")
    
    # Amount-based features
    df = df.withColumn("amount_log", log(col("amount") + 1))
    
    # Time-based features
    if "timestamp" in df.columns:
        df = df.withColumn("hour_of_day", hour(col("timestamp")))
        df = df.withColumn("day_of_week", dayofweek(col("timestamp")))
        df = df.withColumn("month", month(col("timestamp")))
        df = df.withColumn("year", year(col("timestamp")))
        
        # Add cyclical time encoding for better pattern recognition
        from pyspark.sql.functions import sin, cos, lit, radians
        df = df.withColumn("hour_sin", sin(col("hour_of_day") * 2.0 * 3.14159 / 24))
        df = df.withColumn("hour_cos", cos(col("hour_of_day") * 2.0 * 3.14159 / 24))
        df = df.withColumn("day_sin", sin(col("day_of_week") * 2.0 * 3.14159 / 7))
        df = df.withColumn("day_cos", cos(col("day_of_week") * 2.0 * 3.14159 / 7))
        df = df.withColumn("month_sin", sin(col("month") * 2.0 * 3.14159 / 12))
        df = df.withColumn("month_cos", cos(col("month") * 2.0 * 3.14159 / 12))
    
    # Fraud-related features
    if "is_fraud" in df.columns:
        # Convert boolean to integer for ML models
        df = df.withColumn("fraud_label", col("is_fraud").cast("integer"))
    
    # Transaction velocity features
    if "velocity_score" in df.columns:
        # Normalize velocity score
        df = df.withColumn("velocity_score_norm", col("velocity_score") / 100.0)
    
    # Geo-anomaly features
    if "geo_anomaly_score" in df.columns:
        # Bin geo anomaly scores
        df = df.withColumn("geo_anomaly_bin", 
                          when(col("geo_anomaly_score") < 0.3, "low")
                          .when(col("geo_anomaly_score") < 0.7, "medium")
                          .otherwise("high"))
    
    # Device and channel features
    if "device_used" in df.columns and "payment_channel" in df.columns:
        # Create combined feature
        df = df.withColumn("device_channel", 
                          concat(col("device_used"), lit("_"), col("payment_channel")))
    
    # Spending pattern features
    if "spending_deviation_score" in df.columns:
        # Create risk category based on spending deviation
        df = df.withColumn("spending_risk", 
                          when(col("spending_deviation_score") > 1.5, "high")
                          .when(col("spending_deviation_score") > 0.5, "medium")
                          .otherwise("low"))
    
    # Transaction sequence features - crucial for fraud detection patterns
    if "sender_account" in df.columns and "timestamp" in df.columns and "amount" in df.columns:
        from pyspark.sql import Window
        from pyspark.sql.functions import lag, count, avg, stddev, sum, min, max
        
        # Define window for sequence features - transactions ordered by timestamp for each account
        account_window = Window.partitionBy("sender_account").orderBy("timestamp")
        
        # Time between transactions (in seconds)
        df = df.withColumn("time_since_prev_tx", 
                          (col("timestamp").cast("long") - 
                           lag(col("timestamp").cast("long"), 1).over(account_window))/60)
        
        # Amount change from previous transaction
        df = df.withColumn("amount_delta", 
                          col("amount") - lag(col("amount"), 1).over(account_window))
        
        df = df.withColumn("amount_ratio", 
                           when(lag(col("amount"), 1).over(account_window) > 0, 
                                col("amount") / lag(col("amount"), 1).over(account_window))
                           .otherwise(lit(None)))
        
        # Rolling window for recent transaction patterns (last 5 transactions)
        rolling_window_5 = Window.partitionBy("sender_account").orderBy("timestamp").rowsBetween(-5, -1)
        
        # Calculate rolling statistics on recent transactions
        df = df.withColumn("avg_amount_last5", avg(col("amount")).over(rolling_window_5))
        df = df.withColumn("max_amount_last5", max(col("amount")).over(rolling_window_5))
        df = df.withColumn("tx_count_last5", count(col("amount")).over(rolling_window_5))
        
        # Deviation from recent transaction patterns
        df = df.withColumn("amount_deviation_from_avg", 
                          when(col("avg_amount_last5").isNotNull(), 
                               (col("amount") - col("avg_amount_last5")) / col("avg_amount_last5"))
                          .otherwise(0))
        
        # Transaction velocity features
        df = df.withColumn("tx_velocity", col("tx_count_last5") / 5)
    
    logger.info(f"Preprocessing complete. Resulting dataframe has {df.count()} rows and {len(df.columns)} columns")
    return df

def main():
    """
    Main function to execute the data loading and preprocessing pipeline.
    """
    import os
    import argparse
    import logging
    import sys
    
    # Configure logging for real-time output
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Get a logger specific to this module
    logger = logging.getLogger('fraud_detection')
    
    # Only configure if not already configured
    if not logger.handlers:
        # Configure logging to avoid duplicate logs
        logger.setLevel(logging.INFO)
        
        # Check if handlers already exist to prevent duplicates
        file_handler = logging.FileHandler('logs/load_data.log')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        # Only add console handler if not already present
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(console_handler)
    
    # Force stdout to flush immediately
    sys.stdout.reconfigure(line_buffering=True)
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process transaction data for fraud detection')
    parser.add_argument('--input', type=str, default='data/raw/sample_dataset.csv',
                        help='Path to the input CSV file')
    parser.add_argument('--output', type=str, default='data/processed/transactions.parquet',
                        help='Path to save the processed Parquet file')
    
    # GPU configuration arguments
    parser.add_argument('--disable-gpu', action='store_true',
                        help='Disable GPU usage even if available')
    parser.add_argument('--single-gpu', action='store_true',
                        help='Use only a single GPU even if multiple are available')
    parser.add_argument('--memory-growth', action='store_true', default=True,
                        help='Enable memory growth for GPUs to prevent TensorFlow from allocating all memory')
    parser.add_argument('--gpu-memory', type=str, default='2G',
                        help='Amount of GPU memory to allocate for Spark operations (e.g. 2G, 4G)')
    args = parser.parse_args()
    
    # Handle GPU configuration based on command-line arguments
    if args.disable_gpu:
        logger.info("GPU usage disabled by command line argument.")
        try:
            import tensorflow as tf
            tf.config.set_visible_devices([], 'GPU')
        except ImportError:
            pass
        gpu_available = False
    else:
        # Check GPU availability
        gpu_available = False
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            gpu_available = len(gpus) > 0 and not args.disable_gpu
            
            # Configure memory growth if requested
            if args.memory_growth and gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
        except ImportError:
            pass
    
    # Initialize Spark with appropriate GPU configuration
    # Use multi-GPU by default unless single-GPU is specified
    use_multi_gpu = not args.single_gpu
    
    # Initialize Spark with appropriate GPU configuration
    spark = create_spark_session(gpu_available=gpu_available, 
                               multi_gpu=use_multi_gpu,
                               gpu_memory=args.gpu_memory)
    
    # Get absolute paths
    current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raw_path = os.path.join(current_dir, args.input)
    processed_path = os.path.join(current_dir, args.output)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    
    logger.info(f"Using input file: {raw_path}")
    logger.info(f"Using output path: {processed_path}")
    
    # Load and process data
    df_raw = load_data(spark, raw_path)
    
    # Print schema and sample data
    logger.info("Raw data schema:")
    df_raw.printSchema()
    logger.info("Sample data:")
    df_raw.show(5)
    
    # Preprocess data
    df_processed = preprocess_data(df_raw)
    
    # Print processed schema
    logger.info("Processed data schema:")
    df_processed.printSchema()
    
    # Save the processed data for modeling (as Parquet for efficiency)
    logger.info(f"Saving processed data to {processed_path}")
    df_processed.write.mode("overwrite").parquet(processed_path)
    
    logger.info("Data processing complete!")
    
    # Stop Spark session
    spark.stop()

if __name__ == "__main__":
    main()
