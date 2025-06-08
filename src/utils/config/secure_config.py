"""
Secure configuration management for the fraud detection pipeline.
This module provides secure loading of configuration with support for environment variables and secrets.
"""

import os
import yaml
import json
import logging
from pathlib import Path

logger = logging.getLogger("fraud_detection.config")

def load_config(config_path=None):
    """
    Load configuration from a YAML file with environment variable substitution.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    # Default to environment variable or fallback to development config
    if config_path is None:
        config_path = os.environ.get("CONFIG_PATH", "config.yaml")
    
    logger.info(f"Looking for configuration at {config_path}")
    
    try:
        # Try the direct path first
        if os.path.exists(config_path):
            logger.info(f"Loading configuration from {config_path}")
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        # If not found, try in the config directory
        elif os.path.exists(os.path.join('config', os.path.basename(config_path))):
            config_dir_path = os.path.join('config', os.path.basename(config_path))
            logger.info(f"Loading configuration from {config_dir_path}")
            with open(config_dir_path, 'r') as f:
                config = yaml.safe_load(f)
        # If still not found, try relative to the script location
        else:
            script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            script_config_path = os.path.join(script_dir, 'config', os.path.basename(config_path))
            if os.path.exists(script_config_path):
                logger.info(f"Loading configuration from {script_config_path}")
                with open(script_config_path, 'r') as f:
                    config = yaml.safe_load(f)
            else:
                raise FileNotFoundError(f"Configuration file not found at {config_path}, {os.path.join('config', os.path.basename(config_path))}, or {script_config_path}")
        
        
        # Process environment variable substitutions
        config = _process_env_vars(config)
        
        # Load secrets if available
        secrets_path = os.environ.get("SECRETS_PATH")
        if secrets_path and os.path.exists(secrets_path):
            config = _merge_secrets(config, secrets_path)
        
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise

def _process_env_vars(config):
    """
    Process environment variable substitutions in the configuration.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        dict: Configuration with environment variables substituted
    """
    if isinstance(config, dict):
        return {k: _process_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [_process_env_vars(i) for i in config]
    elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
        # Extract environment variable name
        env_var = config[2:-1]
        # Get value with optional default after colon
        if ":" in env_var:
            env_var, default = env_var.split(":", 1)
            return os.environ.get(env_var, default)
        return os.environ.get(env_var, config)
    else:
        return config

def _merge_secrets(config, secrets_path):
    """
    Merge secrets into the configuration.
    
    Args:
        config (dict): Configuration dictionary
        secrets_path (str): Path to the secrets file
        
    Returns:
        dict: Configuration with secrets merged
    """
    try:
        # Determine file type from extension
        if secrets_path.endswith('.json'):
            with open(secrets_path, 'r') as f:
                secrets = json.load(f)
        elif secrets_path.endswith(('.yaml', '.yml')):
            with open(secrets_path, 'r') as f:
                secrets = yaml.safe_load(f)
        else:
            logger.warning(f"Unsupported secrets file format: {secrets_path}")
            return config
        
        # Create a new config with secrets merged in
        return _deep_merge(config, secrets)
    except Exception as e:
        logger.error(f"Error loading secrets: {str(e)}")
        return config

def _deep_merge(dict1, dict2):
    """
    Deep merge two dictionaries.
    
    Args:
        dict1 (dict): First dictionary
        dict2 (dict): Second dictionary
        
    Returns:
        dict: Merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
            
    return result

def save_config(config, output_path):
    """
    Save configuration to a YAML file.
    
    Args:
        config (dict): Configuration dictionary
        output_path (str): Path to save the configuration
    """
    try:
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        logger.info(f"Configuration saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving configuration: {str(e)}")
        raise
