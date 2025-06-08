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
# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline.log'),
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
    
    # Set environment variable to disable output buffering in Python subprocesses
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    
    start_time = time.time()
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
        universal_newlines=True,
        env=env
    )
    
    # Use queue to handle output in a thread-safe way
    import queue
    import threading
    q = queue.Queue()
    
    def enqueue_output(stream, queue_obj, is_error=False):
        for line in iter(stream.readline, ''):
            queue_obj.put((line.strip(), is_error))
        stream.close()
    
    # Start threads to enqueue output
    stdout_thread = threading.Thread(
        target=enqueue_output,
        args=(process.stdout, q),
        daemon=True
    )
    stderr_thread = threading.Thread(
        target=enqueue_output,
        args=(process.stderr, q, True),
        daemon=True
    )
    stdout_thread.start()
    stderr_thread.start()
    
    # Read from queue and log output
    while True:
        try:
            line, is_error = q.get(timeout=0.1)
            if line:
                # Check if this is a Spark progress indicator or TensorFlow initialization message
                if is_error and (
                    line.strip().startswith('[Stage') or 
                    'SparkStringUtils' in line or
                    'metal_plugin' in line or
                    'systemMemory:' in line or
                    'maxCacheSize:' in line or
                    'WARNING: All log messages before absl::InitializeLog()' in line or
                    'pluggable_device_factory' in line or
                    'Metal device set to:' in line or
                    'Created TensorFlow device' in line or
                    line.startswith('WARNING:') or
                    line.startswith('WARN ') or
                    'Using Spark' in line or
                    'Your hostname' in line or
                    'Set SPARK_LOCAL_IP' in line or
                    'Using incubator modules' in line or
                    'Unable to load native-hadoop library' in line or
                    'Setting default log level' in line or
                    'To adjust logging level' in line
                ):
                    # Log these messages at INFO level instead of ERROR
                    logger.info(line)
                elif is_error:
                    logger.error(line)
                else:
                    logger.info(line)
                    print(line, flush=True)
        except queue.Empty:
            # Check if process has terminated and all output has been processed
            if not stdout_thread.is_alive() and not stderr_thread.is_alive() and q.empty():
                break
    
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
    # Get data paths from config
    raw_data_path = config['data']['raw_path']
    processed_data_path = config['data']['processed_path']
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    
    # Get Spark configuration from config or use defaults
    spark_config = config.get('spark', {})
    num_partitions = spark_config.get('num_partitions', 32)
    driver_memory = spark_config.get('driver_memory', '18g')
    executor_memory = spark_config.get('executor_memory', '18g')
    
    # Build command with additional parameters
    command = f"python src/spark_jobs/load_data.py --input {raw_data_path} --output {processed_data_path} "
    command += f"--num-partitions {num_partitions} --driver-memory {driver_memory} --executor-memory {executor_memory}"
    
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
    
    # Add advanced sampling parameters if configured for classification models
    if model_type == 'classification' and 'sampling' in config.get('models', {}):
        sampling_config = config['models']['sampling']
        if 'technique' in sampling_config:
            command += f" --sampling-technique {sampling_config['technique']}"
        if 'ratio' in sampling_config:
            command += f" --sampling-ratio {sampling_config['ratio']}"
        if 'k_neighbors' in sampling_config:
            command += f" --k-neighbors {sampling_config['k_neighbors']}"
    
    # Add custom loss function parameters if configured for classification models
    if model_type == 'classification' and 'loss_function' in config.get('models', {}):
        loss_config = config['models']['loss_function']
        if 'name' in loss_config:
            command += f" --loss-function {loss_config['name']}"
        if 'focal_gamma' in loss_config:
            command += f" --focal-gamma {loss_config['focal_gamma']}"
        if 'focal_alpha' in loss_config:
            command += f" --focal-alpha {loss_config['focal_alpha']}"
        if 'class_weight_ratio' in loss_config:
            command += f" --class-weight-ratio {loss_config['class_weight_ratio']}"
    
    # Add L2 regularization if configured
    if model_type in ['classification', 'autoencoder'] and 'l2_regularization' in config.get('models', {}).get(model_type, {}):
        command += f" --l2-regularization {config['models'][model_type]['l2_regularization']}"
    
    # Add GPU-specific parameters if configured
    if 'gpu' in config and config['gpu']:
        # Add single-GPU flag if multi-GPU is disabled
        if 'multi_gpu' in config['gpu'] and not config['gpu']['multi_gpu']:
            command += f" --single-gpu"
        
        # Add disable-gpu flag if specified
        if 'disable_gpu' in config['gpu'] and config['gpu']['disable_gpu']:
            command += f" --disable-gpu"
        
        # Add batch size multiplier for multi-GPU training if specified
        if 'batch_size_multiplier' in config['gpu']:
            command += f" --batch-size-multiplier {config['gpu']['batch_size_multiplier']}"
        
        # Add memory growth option if specified
        if 'memory_growth' in config['gpu']:
            command += f" --memory-growth"
    
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
    # Use the processed data for testing as well since we're working with a sample dataset
    test_data_path = config['data']['processed_path']
    model_dir = config['models']['output_dir']
    results_dir = config['evaluation']['output_dir']
    
    # Ensure directories exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Find the latest model file with the matching model_type
    import glob
    import json
    
    # First try to find models by examining their parameter files
    # This is more reliable as it checks the actual model type in the parameters
    params_pattern = os.path.join(model_dir, f"*_params.json")
    params_files = glob.glob(params_pattern)
    
    matching_models = []
    
    if params_files:
        # Sort by modification time (newest first)
        params_files.sort(key=os.path.getmtime, reverse=True)
        
        # Check each parameter file to find matching model type
        for params_file in params_files:
            try:
                with open(params_file, 'r') as f:
                    params = json.load(f)
                    # Check if this is the model type we're looking for
                    if params.get('model_type') == model_type:
                        # Convert params filename to model filename
                        # Example: fraud_detection_20250530_105338_params.json -> fraud_detection_20250530_105338_model.keras
                        base_name = os.path.basename(params_file).replace('_params.json', '_model.keras')
                        model_file = os.path.join(model_dir, base_name)
                        if os.path.exists(model_file):
                            matching_models.append(model_file)
            except Exception as e:
                logger.warning(f"Error reading params file {params_file}: {e}")
    
    # If we couldn't find models using params files, fall back to looking for model files directly
    if not matching_models:
        logger.warning(f"Could not find models using parameter files, falling back to filename pattern matching")
        # Look for model files with timestamps
        model_files = glob.glob(os.path.join(model_dir, f"*_model.keras"))
        model_files.sort(key=os.path.getmtime, reverse=True)
        matching_models = model_files
    
    if not matching_models:
        logger.error(f"No model files found for model type: {model_type}")
        return False
    
    # Use the latest matching model
    model_path = matching_models[0]
    logger.info(f"Using latest {model_type} model file: {model_path}")
    
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
    
    # Add GPU-specific parameters if configured
    if 'gpu' in config and config['gpu']:
        # Add single-GPU flag if multi-GPU is disabled
        if 'multi_gpu' in config['gpu'] and not config['gpu']['multi_gpu']:
            command += f" --single-gpu"
        
        # Add disable-gpu flag if specified
        if 'disable_gpu' in config['gpu'] and config['gpu']['disable_gpu']:
            command += f" --disable-gpu"
        
        # Add memory growth option if specified
        if 'memory_growth' in config['gpu']:
            command += f" --memory-growth"
    
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
    
    # GPU configuration arguments
    parser.add_argument('--disable-gpu', action='store_true',
                        help='Disable GPU usage even if available')
    parser.add_argument('--single-gpu', action='store_true',
                        help='Use only a single GPU even if multiple are available')
    parser.add_argument('--batch-size-multiplier', type=int, default=4,
                        help='Multiplier for batch size when using Apple M2 GPU or multiple GPUs')
    parser.add_argument('--memory-growth', action='store_true',
                        help='Enable memory growth for GPUs to prevent TensorFlow from allocating all memory')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Add GPU configuration from command line arguments to config
    if not 'gpu' in config:
        config['gpu'] = {}
    
    # Set multi-GPU as the default, but allow override with single-GPU flag
    if 'multi_gpu' not in config['gpu']:
        config['gpu']['multi_gpu'] = True  # Use all GPUs by default
        logger.info("Multi-GPU training enabled by default")
    
    # Override GPU settings with command line arguments if provided
    if args.disable_gpu:
        config['gpu']['disable_gpu'] = True
        config['gpu']['multi_gpu'] = False  # Disable multi-GPU if GPU is disabled
        logger.info("GPU usage disabled from command line")
    elif args.single_gpu:
        config['gpu']['multi_gpu'] = False
        config['gpu']['disable_gpu'] = False
        logger.info("Single-GPU mode enabled from command line")
    else:
        # Default to multi-GPU if neither disable-gpu nor single-gpu is specified
        config['gpu']['multi_gpu'] = True
        config['gpu']['disable_gpu'] = False
    
    if args.batch_size_multiplier:
        config['gpu']['batch_size_multiplier'] = args.batch_size_multiplier
        logger.info(f"Batch size multiplier set to {args.batch_size_multiplier} from command line")
    
    if args.memory_growth:
        config['gpu']['memory_growth'] = True
        logger.info("GPU memory growth enabled from command line")
    
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
