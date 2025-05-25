"""
Main pipeline script that orchestrates the entire workflow from data processing to model evaluation.
"""
import os
import argparse
import yaml
import subprocess
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('fraud_detection_pipeline')

def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        dict: Configuration parameters
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def run_command(command, description):
    """
    Run a shell command and log the output.
    
    Args:
        command (str): Command to run
        description (str): Description of the command
        
    Returns:
        int: Return code of the command
    """
    logger.info(f"Starting: {description}")
    logger.info(f"Command: {command}")
    
    start_time = time.time()
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
    elapsed_time = time.time() - start_time
    
    if return_code == 0:
        logger.info(f"Completed: {description} in {elapsed_time:.2f} seconds")
    else:
        logger.error(f"Failed: {description} with return code {return_code}")
    
    return return_code

def create_timestamp_dir(base_dir):
    """
    Create a timestamped directory.
    
    Args:
        base_dir (str): Base directory
        
    Returns:
        str: Path to the created directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_path = os.path.join(base_dir, timestamp)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

def run_data_processing(config):
    """
    Run the data processing step.
    
    Args:
        config (dict): Configuration parameters
        
    Returns:
        bool: True if successful, False otherwise
    """
    raw_data_path = config['data']['raw_path']
    processed_data_path = config['data']['processed_path']
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    
    # Build command
    command = f"python src/spark_jobs/load_data.py --input {raw_data_path} --output {processed_data_path}"
    
    # Run command
    return_code = run_command(command, "Data Processing")
    return return_code == 0

def run_model_training(config, model_type):
    """
    Run the model training step.
    
    Args:
        config (dict): Configuration parameters
        model_type (str): Type of model to train ('classification' or 'autoencoder')
        
    Returns:
        bool: True if successful, False otherwise
    """
    processed_data_path = config['data']['processed_path']
    model_dir = config['models']['output_dir']
    
    # Ensure directories exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Build command
    command = (
        f"python src/models/train_model.py "
        f"--data-path {processed_data_path} "
        f"--model-type {model_type} "
        f"--experiment-name {config['mlflow']['experiment_name']} "
        f"--model-dir {model_dir}"
    )
    
    # Run command
    return_code = run_command(command, f"{model_type.capitalize()} Model Training")
    return return_code == 0

def run_model_evaluation(config, model_type):
    """
    Run the model evaluation step.
    
    Args:
        config (dict): Configuration parameters
        model_type (str): Type of model to evaluate ('classification' or 'autoencoder')
        
    Returns:
        bool: True if successful, False otherwise
    """
    test_data_path = config['data']['test_path']
    model_dir = config['models']['output_dir']
    model_path = os.path.join(model_dir, f"{model_type}_model.h5")
    results_dir = config['evaluation']['output_dir']
    
    # Ensure directories exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Build command
    command = (
        f"python src/models/evaluate_model.py "
        f"--model-path {model_path} "
        f"--test-data {test_data_path} "
        f"--model-type {model_type} "
        f"--output-dir {results_dir}"
    )
    
    # Add threshold if specified
    if 'threshold' in config['evaluation'] and config['evaluation']['threshold'] is not None:
        command += f" --threshold {config['evaluation']['threshold']}"
    
    # Add percentile for autoencoder
    if model_type == 'autoencoder' and 'percentile' in config['evaluation']:
        command += f" --percentile {config['evaluation']['percentile']}"
    
    # Run command
    return_code = run_command(command, f"{model_type.capitalize()} Model Evaluation")
    return return_code == 0

def main():
    """
    Main function to run the pipeline.
    """
    parser = argparse.ArgumentParser(description='Run the fraud detection pipeline')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to the configuration file')
    parser.add_argument('--steps', type=str, nargs='+',
                        choices=['all', 'process', 'train', 'evaluate'],
                        default=['all'],
                        help='Pipeline steps to run')
    parser.add_argument('--model-type', type=str,
                        choices=['classification', 'autoencoder', 'both'],
                        default='both',
                        help='Type of model to train/evaluate')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create timestamped run directory
    run_dir = create_timestamp_dir('runs')
    logger.info(f"Pipeline run directory: {run_dir}")
    
    # Determine steps to run
    run_process = 'all' in args.steps or 'process' in args.steps
    run_train = 'all' in args.steps or 'train' in args.steps
    run_evaluate = 'all' in args.steps or 'evaluate' in args.steps
    
    # Determine model types
    model_types = []
    if args.model_type == 'both':
        model_types = ['classification', 'autoencoder']
    else:
        model_types = [args.model_type]
    
    # Run pipeline steps
    success = True
    
    # Step 1: Data Processing
    if run_process:
        logger.info("Starting data processing step")
        success = run_data_processing(config)
        if not success:
            logger.error("Data processing failed. Stopping pipeline.")
            return
    
    # Step 2: Model Training
    if run_train and success:
        logger.info("Starting model training step")
        for model_type in model_types:
            success = run_model_training(config, model_type)
            if not success:
                logger.error(f"{model_type.capitalize()} model training failed. Stopping pipeline.")
                return
    
    # Step 3: Model Evaluation
    if run_evaluate and success:
        logger.info("Starting model evaluation step")
        for model_type in model_types:
            success = run_model_evaluation(config, model_type)
            if not success:
                logger.error(f"{model_type.capitalize()} model evaluation failed.")
    
    # Pipeline completion
    if success:
        logger.info("Pipeline completed successfully!")
    else:
        logger.error("Pipeline completed with errors.")

if __name__ == "__main__":
    main()
