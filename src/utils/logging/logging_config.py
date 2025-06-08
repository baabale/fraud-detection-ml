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
    log_dir = Path("logs")
    if not log_dir.exists():
        log_dir.mkdir(parents=True)
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        # Default configuration
        logging.basicConfig(
            level=default_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("logs/fraud_detection.log"),
                logging.StreamHandler()
            ]
        )
    
    logger = logging.getLogger("fraud_detection")
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
