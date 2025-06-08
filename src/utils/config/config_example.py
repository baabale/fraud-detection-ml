"""
Example script demonstrating how to use the configuration manager.
"""
from src.utils.config_manager import config

def main():
    """Example usage of the configuration manager."""
    # Get data paths
    raw_data_path = config.get_data_path('raw_path')
    processed_data_path = config.get_data_path('processed_path')
    test_data_path = config.get_data_path('test_path')
    
    print(f"Raw data path: {raw_data_path}")
    print(f"Processed data path: {processed_data_path}")
    print(f"Test data path: {test_data_path}")
    
    # Get model configuration
    model_dir = config.get_model_path()
    print(f"Model directory: {model_dir}")
    
    # Get specific model parameters
    classification_params = config.get_model_params('classification')
    autoencoder_params = config.get_model_params('autoencoder')
    
    print(f"Classification model parameters: {classification_params}")
    print(f"Autoencoder model parameters: {autoencoder_params}")
    
    # Get evaluation parameters
    eval_params = config.get_evaluation_params()
    print(f"Evaluation parameters: {eval_params}")
    
    # Get Spark configuration
    spark_config = config.get_spark_config()
    print(f"Spark configuration: {spark_config}")
    
    # Get MLflow configuration
    mlflow_config = config.get_mlflow_config()
    print(f"MLflow configuration: {mlflow_config}")
    
    # Get a specific configuration value using dot notation
    app_name = config.get('spark.app_name')
    print(f"Spark app name: {app_name}")
    
    # Get a non-existent configuration with a default value
    default_value = config.get('non_existent.key', 'default_value')
    print(f"Default value: {default_value}")


if __name__ == "__main__":
    main()
