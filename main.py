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
        # Set environment variable to disable output buffering in Python subprocesses
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=0,  # Unbuffered output
            env=env
        )
        
        # Use a separate thread for each stream to avoid blocking
        import threading
        import queue
        
        def enqueue_output(stream, queue_obj, is_error=False):
            for line in iter(stream.readline, ''):
                queue_obj.put((line.strip(), is_error))
            stream.close()
        
        q = queue.Queue()
        stdout_thread = threading.Thread(target=enqueue_output, args=(process.stdout, q))
        stderr_thread = threading.Thread(target=enqueue_output, args=(process.stderr, q, True))
        stdout_thread.daemon = True
        stderr_thread.daemon = True
        stdout_thread.start()
        stderr_thread.start()
        
        # Read from queue and log in real-time
        while stdout_thread.is_alive() or stderr_thread.is_alive() or not q.empty():
            try:
                line, is_error = q.get(timeout=0.1)
                if is_error:
                    # Check if this is a log line from a subprocess
                    if line.startswith(('20', '19')) and (' - ' in line):
                        # This is already a formatted log line, just print it without re-logging
                        print(line, file=sys.stderr, flush=True)
                    else:
                        logger.error(line)
                        print(line, file=sys.stderr, flush=True)
                else:
                    # Check if this is a log line from a subprocess or training progress message
                    if line.startswith(('20', '19')) and (' - ' in line):
                        # This is already a formatted log line, just print it without re-logging
                        print(line, flush=True)
                    # Check for training progress messages with the time format [X.Xs]
                    elif line.startswith('[') and ']' in line and 'Training in progress' in line:
                        # This is a training progress message, just print it without re-logging
                        print(line, flush=True)
                    else:
                        # Only regular output should be logged
                        logger.info(line)
                        print(line, flush=True)
            except queue.Empty:
                # Check if process has terminated and all output has been processed
                if not stdout_thread.is_alive() and not stderr_thread.is_alive() and q.empty():
                    break
        
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

def run_pipeline(config_path='config.yaml', steps=None, model_type='both', disable_gpu=False, single_gpu=False):
    """
    Run the complete pipeline or specific steps.
    
    Args:
        config_path (str): Path to the configuration file
        steps (list): List of steps to run
        model_type (str): Type of model to train/evaluate
        disable_gpu (bool): Whether to disable GPU usage
        single_gpu (bool): Whether to use only a single GPU
    """
    command = f"python src/pipeline.py --config {config_path}"
    
    if steps:
        command += f" --steps {' '.join(steps)}"
    
    if model_type:
        command += f" --model-type {model_type}"
    
    # Add GPU configuration flags
    if disable_gpu:
        command += " --disable-gpu"
    elif single_gpu:
        command += " --single-gpu"
    
    return run_command(command)

def run_data_processing(config, disable_gpu=False, single_gpu=False):
    """
    Run the data processing step.
    
    Args:
        config (dict): Configuration parameters
        disable_gpu (bool): Whether to disable GPU usage
        single_gpu (bool): Whether to use only a single GPU
    """
    raw_data_path = config.get('data', {}).get('raw_path', 'data/raw/financial_fraud_detection_dataset.csv')
    processed_data_path = config.get('data', {}).get('processed_path', 'data/processed/transactions.parquet')
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    
    command = f"python src/spark_jobs/load_data.py --input {raw_data_path} --output {processed_data_path}"
    
    # Add GPU configuration flags
    if disable_gpu:
        command += " --disable-gpu"
    elif single_gpu:
        command += " --single-gpu"
    
    return run_command(command)

def run_model_training(config, model_type='both', disable_gpu=False, single_gpu=False, batch_size_multiplier=None, memory_growth=False):
    """
    Run the model training step.
    
    Args:
        config (dict): Configuration parameters
        model_type (str): Type of model to train
        disable_gpu (bool): Whether to disable GPU usage
        single_gpu (bool): Whether to use only a single GPU
        batch_size_multiplier (int): Multiplier for batch size when using multiple GPUs
        memory_growth (bool): Enable memory growth for GPUs
    """
    processed_data_path = config.get('data', {}).get('processed_path', 'data/processed/transactions.parquet')
    model_dir = config.get('models', {}).get('output_dir', 'results/models')
    
    # Extract MLflow configuration
    mlflow_config = config.get('mlflow', {})
    experiment_name = mlflow_config.get('experiment_name', 'Fraud_Detection_Experiment')
    mlflow_tracking_uri = mlflow_config.get('tracking_uri', '')
    
    # Check MLflow server connection if a URI is provided
    use_mlflow = True
    if mlflow_tracking_uri and mlflow_tracking_uri.startswith('http'):
        import socket
        import urllib.parse
        try:
            # Parse the URI to get host and port
            parsed_uri = urllib.parse.urlparse(mlflow_tracking_uri)
            host = parsed_uri.hostname
            port = parsed_uri.port or 80
            
            # Try to connect to check if the server is running
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(2.0)  # 2 second timeout
                result = s.connect_ex((host, port))
                if result != 0:
                    print(f"\n⚠️ Warning: MLflow tracking server at {mlflow_tracking_uri} appears to be unavailable.")
                    mlflow_option = input("Select an option:\n1. Continue without MLflow tracking\n2. Use local directory for tracking\n3. Cancel training\nChoice (1-3): ")
                    
                    if mlflow_option == '1':
                        use_mlflow = False
                        print("👉 Continuing without MLflow experiment tracking")
                    elif mlflow_option == '2':
                        local_dir = os.path.join(os.getcwd(), 'mlruns')
                        os.environ['MLFLOW_TRACKING_URI'] = f"file://{local_dir}"
                        print(f"👉 Using local directory for MLflow tracking: {local_dir}")
                    else:  # Option 3 or any other input
                        print("❌ Training cancelled")
                        return 1
        except Exception as e:
            logger.warning(f"Error checking MLflow server: {str(e)}")
            print(f"\n⚠️ Warning: Could not verify MLflow server status. Error: {str(e)}")
            use_mlflow_input = input("Continue with MLflow tracking anyway? (y/[n]): ")
            use_mlflow = use_mlflow_input.lower() == 'y'
    
    # Extract model configuration parameters
    class_config = config.get('models', {}).get('classification', {})
    l2_regularization = class_config.get('l2_regularization', 0.001)
    
    # Extract sampling parameters
    sampling_config = config.get('models', {}).get('sampling', {})
    sampling_technique = sampling_config.get('technique', 'none')
    sampling_ratio = sampling_config.get('ratio', 0.5)
    k_neighbors = sampling_config.get('k_neighbors', 7)
    
    # Extract loss function parameters
    loss_config = config.get('models', {}).get('loss_function', {})
    loss_function = loss_config.get('name', 'binary_crossentropy')
    focal_gamma = loss_config.get('focal_gamma', 4.0)
    focal_alpha = loss_config.get('focal_alpha', 0.3)
    class_weight_ratio = loss_config.get('class_weight_ratio', 5.0)
    
    # Use provided batch_size_multiplier or default to 1
    if batch_size_multiplier is None:
        batch_size_multiplier = 4  # Default to 4 as seen in your command
    
    # Ensure directories exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Set up MLflow tracking if configured and available
    if mlflow_tracking_uri and use_mlflow:
        os.environ['MLFLOW_TRACKING_URI'] = mlflow_tracking_uri
    
    # Build the base command with data path, model directory and type
    command = f"python src/models/train_model.py --data-path {processed_data_path} --model-dir {model_dir} --model-type {model_type}"
    
    # Add experiment name for MLflow tracking (only if using MLflow)
    if use_mlflow:
        command += f" --experiment-name {experiment_name}"
    else:
        command += " --no-mlflow"  # Add a flag to disable MLflow in the script
    
    # Add sampling parameters
    command += f" --sampling-technique {sampling_technique} --sampling-ratio {sampling_ratio} --k-neighbors {k_neighbors}"
    
    # Add loss function parameters
    command += f" --loss-function {loss_function} --focal-gamma {focal_gamma} --focal-alpha {focal_alpha} --class-weight-ratio {class_weight_ratio}"
    
    # Add regularization parameter
    command += f" --l2-regularization {l2_regularization}"
    
    # Add batch size multiplier
    command += f" --batch-size-multiplier {batch_size_multiplier}"
    
    # Add GPU configuration flags
    if disable_gpu:
        command += " --disable-gpu"
    elif single_gpu:
        command += " --single-gpu"
    
    # Add memory growth parameter with a proper boolean value
    memory_growth_value = "true" if memory_growth else "false"
    command += f" --memory-growth {memory_growth_value}"
    
    return run_command(command)

def run_model_evaluation(config, model_type='both', disable_gpu=False, single_gpu=False, memory_growth=False):
    """
    Run the model evaluation step.
    
    Args:
        config (dict): Configuration parameters
        model_type (str): Type of model to evaluate
        disable_gpu (bool): Whether to disable GPU usage
        single_gpu (bool): Whether to use only a single GPU
        memory_growth (bool): Enable memory growth for GPUs
    """
    test_data_path = config.get('data', {}).get('test_path', 'data/processed/test_data.parquet')
    model_dir = config.get('models', {}).get('output_dir', 'results/models')
    results_dir = config.get('evaluation', {}).get('output_dir', 'results')
    
    # Ensure directories exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Base GPU configuration flags
    gpu_flags = ""
    if disable_gpu:
        gpu_flags += " --disable-gpu"
    elif single_gpu:
        gpu_flags += " --single-gpu"
    if memory_growth:
        gpu_flags += " --memory-growth"
    
    if model_type in ['classification', 'both']:
        # Try to find the most recent model file of either format (excluding JSON files)
        classification_models = [f for f in os.listdir(model_dir) 
                               if (f.endswith('.keras') or f.endswith('.h5')) and 
                               (f.startswith('classification_model') or 
                               (f.startswith('fraud_detection_') and not f.startswith('fraud_detection_autoencoder_')))]
        
        if not classification_models:
            logger.warning(f"No classification models found in {model_dir}")
            print(f"⚠️ No classification models found in {model_dir}")
        else:
            # Sort by modification time (newest first)
            classification_models.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
            model_path = os.path.join(model_dir, classification_models[0])
            logger.info(f"Using most recent classification model: {model_path}")
            print(f"Using most recent classification model: {os.path.basename(model_path)}")
            command = (
                f"python src/models/evaluate_model.py "
                f"--model-path {model_path} "
                f"--test-data {test_data_path} "
                f"--model-type classification "
                f"--output-dir {results_dir}"
                f"{gpu_flags}"
            )
            print(f"\nEvaluating classification model...")
            run_command(command)
    
    if model_type in ['autoencoder', 'both']:
        # Try to find the most recent model file of either format (excluding JSON files)
        autoencoder_models = [f for f in os.listdir(model_dir) 
                            if (f.endswith('.keras') or f.endswith('.h5')) and
                            (f.startswith('autoencoder_model') or 
                            f.startswith('fraud_detection_autoencoder_'))]
        
        if not autoencoder_models:
            logger.warning(f"No autoencoder models found in {model_dir}")
            print(f"⚠️ No autoencoder models found in {model_dir}")
        else:
            # Sort by modification time (newest first)
            autoencoder_models.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
            model_path = os.path.join(model_dir, autoencoder_models[0])
            logger.info(f"Using most recent autoencoder model: {model_path}")
            print(f"Using most recent autoencoder model: {os.path.basename(model_path)}")
            command = (
                f"python src/models/evaluate_model.py "
                f"--model-path {model_path} "
                f"--test-data {test_data_path} "
                f"--model-type autoencoder "
                f"--output-dir {results_dir}"
                f"{gpu_flags}"
            )
            print(f"\nEvaluating autoencoder model...")
            run_command(command)

def deploy_model(config, disable_gpu=False, single_gpu=False, memory_growth=False):
    """
    Deploy the model as a REST API.
    
    Args:
        config (dict): Configuration parameters
        disable_gpu (bool): Whether to disable GPU usage
        single_gpu (bool): Whether to use only a single GPU
        memory_growth (bool): Enable memory growth for GPUs
    """
    model_dir = config.get('models', {}).get('model_dir', 'models')
    deployment_dir = config.get('deployment', {}).get('output_dir', 'results/deployment')
    
    # GPU configuration flags
    gpu_flags = ""
    if disable_gpu:
        gpu_flags += " --disable-gpu"
    elif single_gpu:
        gpu_flags += " --single-gpu"
    if memory_growth:
        gpu_flags += " --memory-growth"
    
    # First, save model artifacts for deployment
    command = f"python src/models/save_model_artifacts.py --model-dir {model_dir} --output-dir {deployment_dir}{gpu_flags}"
    run_command(command)
    
    # Then, start the API server
    command = f"python src/models/deploy_model.py --model-dir {deployment_dir} --host 0.0.0.0 --port 8080{gpu_flags}"
    return run_command(command)

def run_streaming(config, disable_gpu=False, single_gpu=False, memory_growth=False):
    """
    Run the streaming fraud detection pipeline.
    
    Args:
        config (dict): Configuration parameters
        disable_gpu (bool): Whether to disable GPU usage
        single_gpu (bool): Whether to use only a single GPU
        memory_growth (bool): Enable memory growth for GPUs
    """
    model_dir = config.get('models', {}).get('model_dir', 'models')
    
    # Start the streaming pipeline with GPU configuration
    command = f"python src/spark_jobs/streaming_fraud_detection.py --model-dir {model_dir}"
    
    # Add GPU configuration flags
    if disable_gpu:
        command += " --disable-gpu"
    elif single_gpu:
        command += " --single-gpu"
    if memory_growth:
        command += " --memory-growth"
    
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

def start_test_api_server():
    """
    Start the API server in test mode for running tests.
    """
    print("\n--- Starting API server for tests ---")
    print("Note: This will start the API server in a new terminal window.")
    print("Please keep it running while executing the tests.")
    
    # Use a background process to start the API server
    command = "python src/api/app.py --test-mode"
    return run_command(command, blocking=False)

def run_tests(test_type='all'):
    """
    Run tests for the fraud detection system.
    
    Args:
        test_type (str): Type of tests to run (unit/integration/api/model/all)
    """
    # Check for required test dependencies
    missing_test_deps = []
    if test_type == 'integration' or test_type == 'all':
        try:
            __import__('prometheus_flask_exporter')
        except ImportError:
            missing_test_deps.append('prometheus_flask_exporter')
    
    # Install missing dependencies if needed
    if missing_test_deps:
        print("\nMissing test dependencies detected:")
        for dep in missing_test_deps:
            print(f"  - {dep}")
        
        install = input("\nWould you like to install missing dependencies? (y/n): ").lower()
        if install == 'y':
            for dep in missing_test_deps:
                print(f"\nInstalling {dep}...")
                run_command(f"pip install {dep}")
            print("\nDependencies installed successfully.")
        else:
            print("\nSkipping dependency installation. Some tests may fail.")
    
    # Run the tests
    if test_type == 'unit' or test_type == 'all':
        print("\n--- Running unit tests ---")
        command = "python -m unittest discover -s tests/unit"
        run_command(command)
    
    if test_type == 'integration' or test_type == 'all':
        print("\n--- Running integration tests ---")
        # Check if API server is running
        print("Note: Integration tests require the API server to be running.")
        command = "python -m unittest discover -s tests/integration"
        run_command(command)
    
    if test_type == 'api' or test_type == 'all':
        print("\n--- Running API tests ---")
        # Check if API server is running
        print("Note: API tests require the API server to be running on port 8080.")
        command = "python test_api.py"
        run_command(command)
    
    if test_type == 'model' or test_type == 'all':
        print("\n--- Running model debug tests ---")
        command = "python debug_model.py"
        run_command(command)
        
    return True

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
    
    # Check for GPU availability
    gpu_available = False
    num_gpus = 0
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        gpu_available = len(gpus) > 0
        num_gpus = len(gpus)
        if gpu_available:
            print(f"\n🚀 {num_gpus} GPU(s) detected and available for acceleration")
            for i, gpu in enumerate(gpus):
                print(f"  - GPU {i}: {gpu.name}")
        else:
            print("\n⚠️ No GPUs detected. Running in CPU-only mode.")
    except ImportError:
        print("\n⚠️ TensorFlow not available. Running in CPU-only mode.")
    except Exception as e:
        print(f"\n⚠️ Error detecting GPUs: {str(e)}. Running in CPU-only mode.")
    
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
        print("10. Run tests")
        print("0. Exit")
        
        choice = input("\nEnter your choice (0-10): ")
        
        if choice == '0':
            print("\nExiting the system. Goodbye!")
            break
        
        elif choice == '1':
            print("\n--- Running the complete pipeline ---")
            steps = input("Which steps to run? (all/process/train/evaluate/[all]): ") or "all"
            model_type = input("Which model type? (classification/autoencoder/both/[both]): ") or "both"
            
            # GPU configuration options
            if gpu_available:
                print(f"\nGPU Configuration:")
                disable_gpu = input(f"Disable GPU acceleration? (y/[n]): ").lower() == 'y'
                single_gpu = False if disable_gpu else (input(f"Use only a single GPU? ({num_gpus} available) (y/[n]): ").lower() == 'y' if num_gpus > 1 else False)
                memory_growth = False if disable_gpu else (input(f"Enable memory growth for GPUs? ([y]/n): ").lower() != 'n')
                
                if not disable_gpu:
                    print(f"\n🚀 Running with {'single GPU' if single_gpu else 'all GPUs'} acceleration")
            else:
                disable_gpu = True
                single_gpu = False
                memory_growth = False
            
            steps_list = steps.split() if steps != "all" else ["all"]
            run_pipeline(config_path, steps_list, model_type, disable_gpu, single_gpu)
        
        elif choice == '2':
            print("\n--- Processing the data ---")
            
            # GPU configuration options
            if gpu_available:
                print(f"\nGPU Configuration:")
                disable_gpu = input(f"Disable GPU acceleration? (y/[n]): ").lower() == 'y'
                single_gpu = False if disable_gpu else (input(f"Use only a single GPU? ({num_gpus} available) (y/[n]): ").lower() == 'y' if num_gpus > 1 else False)
                
                if not disable_gpu:
                    print(f"\n🚀 Running with {'single GPU' if single_gpu else 'all GPUs'} acceleration")
            else:
                disable_gpu = True
                single_gpu = False
            
            run_data_processing(config, disable_gpu, single_gpu)
        
        elif choice == '3':
            print("\n--- Training models ---")
            model_type = input("Which model type? (classification/autoencoder/both/[both]): ") or "both"
            
            # GPU configuration options
            if gpu_available:
                print(f"\nGPU Configuration:")
                disable_gpu = input(f"Disable GPU acceleration? (y/[n]): ").lower() == 'y'
                single_gpu = False if disable_gpu else (input(f"Use only a single GPU? ({num_gpus} available) (y/[n]): ").lower() == 'y' if num_gpus > 1 else False)
                memory_growth = False if disable_gpu else (input(f"Enable memory growth for GPUs? ([y]/n): ").lower() != 'n')
                
                # Advanced GPU options for training
                batch_size_multiplier = None
                if not disable_gpu and not single_gpu and num_gpus > 1:
                    use_batch_multiplier = input(f"Adjust batch size for multi-GPU training? (y/[n]): ").lower() == 'y'
                    if use_batch_multiplier:
                        try:
                            batch_size_multiplier = int(input(f"Batch size multiplier (default is {num_gpus}): ") or num_gpus)
                        except ValueError:
                            print("Invalid input. Using default multiplier.")
                            batch_size_multiplier = num_gpus
                
                if not disable_gpu:
                    print(f"\n🚀 Training with {'single GPU' if single_gpu else 'all GPUs'} acceleration")
                    if batch_size_multiplier:
                        print(f"Batch size will be multiplied by {batch_size_multiplier}")
            else:
                disable_gpu = True
                single_gpu = False
                memory_growth = False
                batch_size_multiplier = None
            
            run_model_training(config, model_type, disable_gpu, single_gpu, batch_size_multiplier, memory_growth)
        
        elif choice == '4':
            print("\n--- Evaluating models ---")
            model_type = input("Which model type? (classification/autoencoder/both/[both]): ") or "both"
            
            # GPU configuration options
            if gpu_available:
                print(f"\nGPU Configuration:")
                disable_gpu = input(f"Disable GPU acceleration? (y/[n]): ").lower() == 'y'
                single_gpu = False if disable_gpu else (input(f"Use only a single GPU? ({num_gpus} available) (y/[n]): ").lower() == 'y' if num_gpus > 1 else False)
                memory_growth = False if disable_gpu else (input(f"Enable memory growth for GPUs? ([y]/n): ").lower() != 'n')
                
                if not disable_gpu:
                    print(f"\n🚀 Evaluating with {'single GPU' if single_gpu else 'all GPUs'} acceleration")
            else:
                disable_gpu = True
                single_gpu = False
                memory_growth = False
            
            run_model_evaluation(config, model_type, disable_gpu, single_gpu, memory_growth)
        
        elif choice == '5':
            print("\n--- Deploying models as API ---")
            
            # GPU configuration options
            if gpu_available:
                print(f"\nGPU Configuration:")
                disable_gpu = input(f"Disable GPU acceleration? (y/[n]): ").lower() == 'y'
                single_gpu = False if disable_gpu else (input(f"Use only a single GPU? ({num_gpus} available) (y/[n]): ").lower() == 'y' if num_gpus > 1 else False)
                memory_growth = False if disable_gpu else (input(f"Enable memory growth for GPUs? ([y]/n): ").lower() != 'n')
                
                if not disable_gpu:
                    print(f"\n🚀 Deploying with {'single GPU' if single_gpu else 'all GPUs'} acceleration")
            else:
                disable_gpu = True
                single_gpu = False
                memory_growth = False
            
            deploy_model(config, disable_gpu, single_gpu, memory_growth)
        
        elif choice == '6':
            print("\n--- Running streaming fraud detection ---")
            
            # GPU configuration options
            if gpu_available:
                print(f"\nGPU Configuration:")
                disable_gpu = input(f"Disable GPU acceleration? (y/[n]): ").lower() == 'y'
                single_gpu = False if disable_gpu else (input(f"Use only a single GPU? ({num_gpus} available) (y/[n]): ").lower() == 'y' if num_gpus > 1 else False)
                memory_growth = False if disable_gpu else (input(f"Enable memory growth for GPUs? ([y]/n): ").lower() != 'n')
                
                if not disable_gpu:
                    print(f"\n🚀 Running streaming with {'single GPU' if single_gpu else 'all GPUs'} acceleration")
            else:
                disable_gpu = True
                single_gpu = False
                memory_growth = False
            
            run_streaming(config, disable_gpu, single_gpu, memory_growth)
        
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
        
        elif choice == '10':
            print("\n--- Running tests ---")
            test_type = input("Which tests to run? (unit/integration/api/model/[all]): ") or "all"
            
            # For API and integration tests, ask if the user wants to start the API server
            if test_type in ['api', 'integration', 'all']:
                start_api = input("Do you want to start the API server for testing? (y/[n]): ").lower() == 'y'
                if start_api:
                    start_test_api_server()
                    print("\nAPI server started. Please wait a few seconds for it to initialize...")
                    import time
                    time.sleep(5)  # Give the server some time to start
            
            run_tests(test_type)
        
        else:
            print("\nInvalid choice. Please try again.")
        
        # Pause before showing the menu again
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
