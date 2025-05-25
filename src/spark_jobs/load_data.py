"""
PySpark script for loading and preprocessing banking transaction data for fraud detection.
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, log, hour, dayofweek, month, year, datediff, current_date
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
    # Read raw transaction data (assuming CSV with header and inferSchema)
    df = spark.read.csv(file_path, header=True, inferSchema=True)
    print(f"Loaded {df.count()} transactions from {file_path}")
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
    
    # Drop rows with null values in critical columns
    df = df.dropna(subset=["amount", "timestamp", "sender"])
    
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
    
    # Account age feature (if account_open_date exists)
    if "account_open_date" in df.columns:
        df = df.withColumn("account_age_days", 
                          datediff(current_date(), col("account_open_date")))
    
    # Add more preprocessing as required
    
    print(f"Preprocessing complete. Resulting dataframe has {df.count()} rows and {len(df.columns)} columns")
    return df

def main():
    """
    Main function to execute the data loading and preprocessing pipeline.
    """
    # Initialize Spark
    spark = create_spark_session()
    
    # Define file paths
    raw_path = "../../data/raw/transactions.csv"
    processed_path = "../../data/processed/transactions.parquet"
    
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
