"""
PySpark script for loading and preprocessing banking transaction data for fraud detection.
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, log, hour, dayofweek, month, year, datediff, current_date, \
    when, lit, concat
from pyspark.sql.types import TimestampType

def create_spark_session(app_name="FraudDetectionPreprocessing"):
    """
    Create and return a Spark session.
    
    Args:
        app_name (str): Name of the Spark application
        
    Returns:
        SparkSession: Configured Spark session
    """
    return SparkSession.builder \
        .appName(app_name) \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()

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
    # Data cleaning
    print("Starting data preprocessing...")
    
    # Print column names for debugging
    print("Available columns:", df.columns)
    
    # Drop rows with null values in critical columns
    df = df.dropna(subset=["amount", "timestamp", "sender_account"])
    
    # Convert timestamp column to proper type (if needed)
    if "timestamp" in df.columns:
        df = df.withColumn("timestamp", col("timestamp").cast(TimestampType()))
    
    # Feature engineering
    print("Performing feature engineering...")
    
    # Amount-based features
    df = df.withColumn("amount_log", log(col("amount") + 1))
    
    # Time-based features
    if "timestamp" in df.columns:
        df = df.withColumn("hour_of_day", hour(col("timestamp")))
        df = df.withColumn("day_of_week", dayofweek(col("timestamp")))
        df = df.withColumn("month", month(col("timestamp")))
        df = df.withColumn("year", year(col("timestamp")))
    
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
    
    print(f"Preprocessing complete. Resulting dataframe has {df.count()} rows and {len(df.columns)} columns")
    return df

def main():
    """
    Main function to execute the data loading and preprocessing pipeline.
    """
    import os
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process transaction data for fraud detection')
    parser.add_argument('--input', type=str, default='data/raw/sample_dataset.csv',  # CHANGED: Using sample dataset
                        help='Path to the input CSV file')
    parser.add_argument('--output', type=str, default='data/processed/transactions.parquet',
                        help='Path to save the processed Parquet file')
    args = parser.parse_args()
    
    # Initialize Spark
    spark = create_spark_session()
    
    # Get absolute paths
    current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raw_path = os.path.join(current_dir, args.input)
    processed_path = os.path.join(current_dir, args.output)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    
    print(f"Using input file: {raw_path}")
    print(f"Using output path: {processed_path}")
    
    # Load and process data
    df_raw = load_data(spark, raw_path)
    
    # Print schema and sample data
    print("Raw data schema:")
    df_raw.printSchema()
    print("Sample data:")
    df_raw.show(5)
    
    # Preprocess data
    df_processed = preprocess_data(df_raw)
    
    # Print processed schema
    print("Processed data schema:")
    df_processed.printSchema()
    
    # Save the processed data for modeling (as Parquet for efficiency)
    print(f"Saving processed data to {processed_path}")
    df_processed.write.mode("overwrite").parquet(processed_path)
    
    print("Data processing complete!")
    
    # Stop Spark session
    spark.stop()

if __name__ == "__main__":
    main()
