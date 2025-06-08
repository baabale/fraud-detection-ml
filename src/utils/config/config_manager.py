"""
Configuration management utility for the fraud detection system.
Provides a centralized way to access configuration settings.
"""
import os
import sys
import yaml
from pathlib import Path

# Add the project root to the path to ensure imports work from any directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class ConfigManager:
    """
    Configuration manager for the fraud detection system.
    Loads configuration from a YAML file and provides access to settings.
    """
    _instance = None
    
    def __new__(cls, config_path=None):
        """Singleton pattern to ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config_path=None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path (str, optional): Path to the configuration file.
                If None, uses the default config.yaml in the project root.
        """
        if self._initialized:
            return
            
        # Get project root directory
        self.project_root = self._get_project_root()
        
        # Load configuration
        if config_path is None:
            config_path = 'config.yaml'
        
        # Try different possible locations for the config file
        config_locations = [
            config_path,  # Direct path
            os.path.join('config', os.path.basename(config_path)),  # In config directory
            os.path.join(self.project_root, 'config', os.path.basename(config_path)),  # Project root config directory
            os.path.join(self.project_root, os.path.basename(config_path))  # Project root
        ]
        
        for location in config_locations:
            if os.path.exists(location):
                with open(location, 'r') as f:
                    self.config = yaml.safe_load(f)
                logger.info(f"Configuration loaded from {location}")
                break
        else:
            raise FileNotFoundError(f"Configuration file not found in any of these locations: {config_locations}")
        
            
        self._initialized = True
    
    def _get_project_root(self):
        """Get the absolute path to the project root directory."""
        # This file is in src/utils, so we need to go up two levels
        current_file = Path(__file__).resolve()
        return str(current_file.parent.parent.parent)
    
    def get_absolute_path(self, relative_path):
        """
        Convert a relative path to an absolute path based on the project root.
        
        Args:
            relative_path (str): Relative path from the project root.
            
        Returns:
            str: Absolute path.
        """
        return os.path.join(self.project_root, relative_path)
    
    def get(self, key_path, default=None):
        """
        Get a configuration value using a dot-separated path.
        
        Args:
            key_path (str): Dot-separated path to the configuration value.
                Example: 'data.raw_path'
            default: Default value to return if the key is not found.
            
        Returns:
            The configuration value, or the default if not found.
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_data_path(self, key, absolute=True):
        """
        Get a data path from the configuration.
        
        Args:
            key (str): Key for the data path (e.g., 'raw_path').
            absolute (bool): If True, returns an absolute path.
            
        Returns:
            str: The data path.
        """
        path = self.get(f'data.{key}')
        if path and absolute:
            return self.get_absolute_path(path)
        return path
    
    def get_model_path(self, key=None, absolute=True):
        """
        Get a model path from the configuration.
        
        Args:
            key (str, optional): Specific model key. If None, returns the model output directory.
            absolute (bool): If True, returns an absolute path.
            
        Returns:
            str: The model path.
        """
        if key:
            path = self.get(f'models.{key}')
        else:
            path = self.get('models.output_dir')
            
        if path and absolute:
            return self.get_absolute_path(path)
        return path
    
    def get_model_params(self, model_type):
        """
        Get model parameters for a specific model type.
        
        Args:
            model_type (str): Type of model (e.g., 'classification', 'autoencoder').
            
        Returns:
            dict: Model parameters.
        """
        return self.get(f'models.{model_type}', {})
    
    def get_evaluation_params(self):
        """
        Get evaluation parameters.
        
        Returns:
            dict: Evaluation parameters.
        """
        return self.get('evaluation', {})
    
    def get_spark_config(self):
        """
        Get Spark configuration.
        
        Returns:
            dict: Spark configuration.
        """
        return self.get('spark', {})
    
    def get_mlflow_config(self):
        """
        Get MLflow configuration.
        
        Returns:
            dict: MLflow configuration.
        """
        return self.get('mlflow', {})


# Create a singleton instance for easy import
config = ConfigManager()
