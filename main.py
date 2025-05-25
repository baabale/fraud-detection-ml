#!/usr/bin/env python
"""
Main entry point for the Anomaly-Based Fraud Detection System.
This script provides a unified interface to run different parts of the project.
"""
import os
import argparse
import subprocess
import yaml
import logging
import sys
from datetime import datetime

# Check for required dependencies
def check_dependencies():
    missing_deps = []
    optional_deps = {
        'tensorflow': "Deep learning models will not be available",
        'pyspark': "Distributed data processing will not be available",
        'mlflow': "Experiment tracking will not be available"
    }
    
    for dep, message in optional_deps.items():
        try:
            __import__(dep)
        except ImportError:
            missing_deps.append(f"{dep}: {message}")
    
    return missing_deps

# Configure logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/fraud_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('fraud_detection')

def load_config(config_path='config.yaml'):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        dict: Configuration parameters
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        return {}

def run_command(command):
    """
    Run a shell command and log the output.
    
    Args:
        command (str): Command to run
        
    Returns:
        int: Return code of the command
    """
    logger.info(f"Running command: {command}")
    
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Stream output to logs
        for stdout_line in iter(process.stdout.readline, ""):
            if stdout_line:
                logger.info(stdout_line.strip())
        
        for stderr_line in iter(process.stderr.readline, ""):
            if stderr_line:
                logger.error(stderr_line.strip())
        
        # Wait for process to complete
        return_code = process.wait()
        
        if return_code == 0:
            logger.info(f"Command completed successfully")
        else:
            logger.error(f"Command failed with return code {return_code}")
        
        return return_code
    
    except Exception as e:
        logger.error(f"Error executing command: {str(e)}")
        return 1

def run_pipeline(config_path='config.yaml', steps=None, model_type='both'):
    """
    Run the complete pipeline or specific steps.
    
    Args:
        config_path (str): Path to the configuration file
        steps (list): List of steps to run
        model_type (str): Type of model to train/evaluate
    """
    command = f"python src/pipeline.py --config {config_path}"
    
    if steps:
        command += f" --steps {' '.join(steps)}"
    
    if model_type:
        command += f" --model-type {model_type}"
    
    return run_command(command)

def run_data_processing(config):
    """
    Run the data processing step.
    
    Args:
        config (dict): Configuration parameters
    """
    raw_data_path = config.get('data', {}).get('raw_path', 'data/raw/financial_fraud_detection_dataset.csv')
    processed_data_path = config.get('data', {}).get('processed_path', 'data/processed/transactions.parquet')
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    
    command = f"python src/spark_jobs/load_data.py --input {raw_data_path} --output {processed_data_path}"
    return run_command(command)

def run_model_training(config, model_type='both'):
    """
    Run the model training step.
    
    Args:
        config (dict): Configuration parameters
        model_type (str): Type of model to train
    """
    processed_data_path = config.get('data', {}).get('processed_path', 'data/processed/transactions.parquet')
    model_dir = config.get('models', {}).get('output_dir', 'results/models')
    experiment_name = config.get('mlflow', {}).get('experiment_name', 'Fraud_Detection_Experiment')
    
    # Ensure directories exist
    os.makedirs(model_dir, exist_ok=True)
    
    command = (
        f"python src/models/train_model.py "
        f"--data-path {processed_data_path} "
        f"--model-type {model_type} "
        f"--experiment-name {experiment_name} "
        f"--model-dir {model_dir}"
    )
    
    return run_command(command)

def run_model_evaluation(config, model_type='both'):
    """
    Run the model evaluation step.
    
    Args:
        config (dict): Configuration parameters
        model_type (str): Type of model to evaluate
    """
    test_data_path = config.get('data', {}).get('test_path', 'data/processed/test_data.parquet')
    model_dir = config.get('models', {}).get('output_dir', 'results/models')
    results_dir = config.get('evaluation', {}).get('output_dir', 'results')
    
    # Ensure directories exist
    os.makedirs(results_dir, exist_ok=True)
    
    if model_type in ['classification', 'both']:
        model_path = os.path.join(model_dir, 'classification_model.h5')
        if os.path.exists(model_path):
            command = (
                f"python src/models/evaluate_model.py "
                f"--model-path {model_path} "
                f"--test-data {test_data_path} "
                f"--model-type classification "
                f"--output-dir {results_dir}"
            )
            run_command(command)
    
    if model_type in ['autoencoder', 'both']:
        model_path = os.path.join(model_dir, 'autoencoder_model.h5')
        if os.path.exists(model_path):
            command = (
                f"python src/models/evaluate_model.py "
                f"--model-path {model_path} "
                f"--test-data {test_data_path} "
                f"--model-type autoencoder "
                f"--output-dir {results_dir}"
            )
            run_command(command)

def deploy_model(config):
    """
    Deploy the model as a REST API.
    
    Args:
        config (dict): Configuration parameters
    """
    model_dir = config.get('models', {}).get('output_dir', 'results/models')
    
    # First, save model artifacts for deployment
    command = f"python src/models/save_model_artifacts.py --output-dir results/deployment"
    run_command(command)
    
    # Then, start the API server
    command = f"python src/models/deploy_model.py --model-dir results/deployment --host 0.0.0.0 --port 5000"
    return run_command(command)

def run_streaming(config):
    """
    Run the streaming fraud detection pipeline.
    
    Args:
        config (dict): Configuration parameters
    """
    model_dir = config.get('models', {}).get('output_dir', 'results/deployment')
    
    # Start the streaming pipeline
    command = f"python src/spark_jobs/streaming_fraud_detection.py --model-dir {model_dir}"
    return run_command(command)

def generate_test_data(num_transactions=1000, format='json'):
    """
    Generate synthetic test data.
    
    Args:
        num_transactions (int): Number of transactions to generate
        format (str): Output format
    """
    command = f"python src/utils/generate_test_data.py --num-transactions {num_transactions} --format {format}"
    return run_command(command)

def monitor_model(config):
    """
    Run model monitoring to detect data drift and performance degradation.
    
    Args:
        config (dict): Configuration parameters
    """
    reference_data = config.get('data', {}).get('processed_path', 'data/processed/transactions.parquet')
    production_data = config.get('data', {}).get('test_path', 'data/test/transactions.parquet')
    model_dir = 'results/deployment'
    output_dir = 'results/monitoring'
    
    command = (
        f"python src/models/model_monitoring.py "
        f"--reference-data {reference_data} "
        f"--production-data {production_data} "
        f"--model-dir {model_dir} "
        f"--output-dir {output_dir}"
    )
    
    return run_command(command)

def main():
    """
    Main function that provides an interactive menu to run different parts of the system.
    """
    print("\n" + "="*60)
    print("   ANOMALY-BASED FRAUD DETECTION SYSTEM")
    print("="*60 + "\n")
    
    # Check for missing dependencies
    missing_deps = check_dependencies()
    if missing_deps:
        print("\nWARNING: Some dependencies are missing. Certain features may be limited.")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\nYou can install missing dependencies with: pip install <package_name>")
        print("\nPress Enter to continue with limited functionality...")
        try:
            input()
        except EOFError:
            # Handle case where input() is not available (e.g., in non-interactive environments)
            print("\nContinuing with limited functionality...")
    
    # Load configuration
    config_path = 'config.yaml'
    config = load_config(config_path)
    
    while True:
        print("\nWhat would you like to do?")
        print("\n1. Run the complete pipeline")
        print("2. Process the data")
        print("3. Train models")
        print("4. Evaluate models")
        print("5. Deploy models as API")
        print("6. Run streaming fraud detection")
        print("7. Generate synthetic test data")
        print("8. Monitor model performance")
        print("9. Launch Jupyter notebooks")
        print("0. Exit")
        
        choice = input("\nEnter your choice (0-9): ")
        
        if choice == '0':
            print("\nExiting the system. Goodbye!")
            break
        
        elif choice == '1':
            print("\n--- Running the complete pipeline ---")
            steps = input("Which steps to run? (all/process/train/evaluate/[all]): ") or "all"
            model_type = input("Which model type? (classification/autoencoder/both/[both]): ") or "both"
            
            steps_list = steps.split() if steps != "all" else ["all"]
            run_pipeline(config_path, steps_list, model_type)
        
        elif choice == '2':
            print("\n--- Processing the data ---")
            run_data_processing(config)
        
        elif choice == '3':
            print("\n--- Training models ---")
            model_type = input("Which model type? (classification/autoencoder/both/[both]): ") or "both"
            run_model_training(config, model_type)
        
        elif choice == '4':
            print("\n--- Evaluating models ---")
            model_type = input("Which model type? (classification/autoencoder/both/[both]): ") or "both"
            run_model_evaluation(config, model_type)
        
        elif choice == '5':
            print("\n--- Deploying models as API ---")
            deploy_model(config)
        
        elif choice == '6':
            print("\n--- Running streaming fraud detection ---")
            run_streaming(config)
        
        elif choice == '7':
            print("\n--- Generating synthetic test data ---")
            num_transactions = input("Number of transactions to generate [1000]: ") or "1000"
            format_type = input("Output format (json/csv/parquet/stream/[json]): ") or "json"
            generate_test_data(int(num_transactions), format_type)
        
        elif choice == '8':
            print("\n--- Monitoring model performance ---")
            monitor_model(config)
        
        elif choice == '9':
            print("\n--- Launching Jupyter notebooks ---")
            run_command("jupyter notebook notebooks/")
        
        else:
            print("\nInvalid choice. Please try again.")
        
        # Pause before showing the menu again
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
