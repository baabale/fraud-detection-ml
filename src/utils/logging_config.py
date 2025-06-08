"""
Logging configuration for the fraud detection pipeline.
This module provides a centralized logging configuration for the entire application.
"""

import os
import logging
import logging.config
import yaml
from pathlib import Path

def setup_logging(config_path=None, default_level=logging.INFO):
    """
    Setup logging configuration from a YAML file or default configuration.

    Args:
        config_path (str): Path to the logging configuration file
        default_level (int): Default logging level if config file is not found

    Returns:
        logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    if not log_dir.exists():
        log_dir.mkdir(parents=True)

    # Configure TensorFlow logging - completely disable it
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=INFO, 2=WARNING, 3=ERROR
    # Disable TensorFlow's internal logging
    os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
    # Disable TensorFlow's direct stderr output
    os.environ['TF_DISABLE_STDERR_OUTPUT'] = '1'

    if config_path and os.path.exists(config_path):
        with open(config_path, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        # Default configuration with custom formatters
        logging.basicConfig(
            level=default_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("logs/fraud_detection.log"),
                logging.StreamHandler()
            ]
        )

        # Configure TensorFlow logger to be completely silent
        tf_logger = logging.getLogger('tensorflow')
        tf_logger.setLevel(logging.ERROR)  # Only show errors, not warnings
        tf_logger.propagate = False  # Don't propagate to parent loggers

        # Configure absl logger (used by TensorFlow) to be completely silent
        absl_logger = logging.getLogger('absl')
        absl_logger.setLevel(logging.ERROR)  # Only show errors, not warnings
        absl_logger.propagate = False  # Don't propagate to parent loggers

    # Get the main application logger
    logger = logging.getLogger("fraud_detection")

    # Add a filter to clean up GPU-related messages and TensorFlow training progress
    class GPUMessageFilter(logging.Filter):
        def filter(self, record):
            # Filter out common noisy GPU messages and TensorFlow training progress
            noisy_messages = [
                'Created TensorFlow device',
                'Metal device set to',
                'pluggable_device_factory',
                'systemMemory:',
                'maxCacheSize:',
                'WARNING: All log messages before absl::InitializeLog()',
                'Using Spark',
                'Your hostname',
                'Set SPARK_LOCAL_IP',
                'Using incubator modules',
                'Unable to load native-hadoop library',
                'Setting default log level',
                'To adjust logging level',
                '/step - ',      # TensorFlow training step updates
                'ETA: ',         # TensorFlow ETA messages
                'loss: ',        # TensorFlow loss updates
                '[======',       # Progress bar
                '[>>>>>',        # Progress bar
                '[=====',        # Progress bar
                '/epoch',        # Epoch indicators
                'Training in progress'  # Our custom training progress messages
            ]
            # Filter out any messages containing these patterns
            if any(msg in record.getMessage() for msg in noisy_messages):
                return False
                
            # Also filter out any messages that match our training progress timestamp format
            # This catches all messages like "[10.1s] Training in progress..."
            msg = record.getMessage()
            if msg.startswith('[') and '.s]' in msg and 'Training in progress' in msg:
                return False
                
            return True

    # Apply the filter to the root logger
    logging.getLogger().addFilter(GPUMessageFilter())

    logger.info("Logging configured successfully")
    return logger

def get_logger(name):
    """
    Get a logger with the given name.

    Args:
        name (str): Logger name, typically the module name

    Returns:
        logger: Logger instance
    """
    return logging.getLogger(f"fraud_detection.{name}")
