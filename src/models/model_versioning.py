"""
Model versioning system for the fraud detection pipeline.
This module provides functionality for versioning, registering, and retrieving models.
"""

import os
import json
import shutil
import logging
import datetime
import mlflow
from pathlib import Path

# Import project modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logging_config import get_logger

# Configure logging
logger = get_logger("model_versioning")

class ModelRegistry:
    """
    Model registry for managing model versions.
    """
    
    def __init__(self, registry_dir, mlflow_tracking_uri=None):
        """
        Initialize the model registry.
        
        Args:
            registry_dir (str): Directory for the model registry
            mlflow_tracking_uri (str): MLflow tracking URI
        """
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.registry_dir / "registry.json"
        
        # Initialize registry if it doesn't exist
        if not self.registry_file.exists():
            self._initialize_registry()
        
        # Set MLflow tracking URI if provided
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    def _initialize_registry(self):
        """Initialize an empty registry."""
        registry = {
            "models": {},
            "last_updated": datetime.datetime.now().isoformat()
        }
        self._save_registry(registry)
    
    def _load_registry(self):
        """Load the registry from file."""
        try:
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading registry: {str(e)}")
            return {"models": {}, "last_updated": datetime.datetime.now().isoformat()}
    
    def _save_registry(self, registry):
        """Save the registry to file."""
        try:
            registry["last_updated"] = datetime.datetime.now().isoformat()
            with open(self.registry_file, 'w') as f:
                json.dump(registry, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving registry: {str(e)}")
    
    def register_model(self, model_name, model_path, model_type, metrics, metadata=None, 
                      mlflow_run_id=None, production=False):
        """
        Register a model in the registry.
        
        Args:
            model_name (str): Name of the model
            model_path (str): Path to the model files
            model_type (str): Type of model (classification, autoencoder, etc.)
            metrics (dict): Performance metrics
            metadata (dict): Additional metadata
            mlflow_run_id (str): MLflow run ID
            production (bool): Whether to mark as production model
            
        Returns:
            str: Version ID of the registered model
        """
        registry = self._load_registry()
        
        # Create model entry if it doesn't exist
        if model_name not in registry["models"]:
            registry["models"][model_name] = {
                "versions": [],
                "production_version": None,
                "staging_version": None
            }
        
        # Generate version ID
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        version_id = f"{model_name}_{timestamp}"
        
        # Create version directory
        version_dir = self.registry_dir / version_id
        version_dir.mkdir(exist_ok=True)
        
        # Copy model files to version directory
        try:
            if os.path.isdir(model_path):
                shutil.copytree(model_path, version_dir / "model", dirs_exist_ok=True)
            else:
                shutil.copy2(model_path, version_dir / "model")
        except Exception as e:
            logger.error(f"Error copying model files: {str(e)}")
            return None
        
        # Create version info
        version_info = {
            "id": version_id,
            "created_at": datetime.datetime.now().isoformat(),
            "model_type": model_type,
            "metrics": metrics,
            "metadata": metadata or {},
            "mlflow_run_id": mlflow_run_id
        }
        
        # Save version info
        with open(version_dir / "info.json", 'w') as f:
            json.dump(version_info, f, indent=2)
        
        # Add to registry
        registry["models"][model_name]["versions"].append(version_info)
        
        # Set as production or staging
        if production:
            registry["models"][model_name]["production_version"] = version_id
        else:
            registry["models"][model_name]["staging_version"] = version_id
        
        # Save registry
        self._save_registry(registry)
        
        logger.info(f"Registered model {model_name} version {version_id}")
        return version_id
    
    def get_model_version(self, model_name, version_id=None, stage=None):
        """
        Get a specific model version.
        
        Args:
            model_name (str): Name of the model
            version_id (str): Version ID (optional)
            stage (str): Stage to retrieve (production, staging) (optional)
            
        Returns:
            dict: Model version information
        """
        registry = self._load_registry()
        
        if model_name not in registry["models"]:
            logger.error(f"Model {model_name} not found in registry")
            return None
        
        model_info = registry["models"][model_name]
        
        # Determine which version to retrieve
        if version_id:
            # Find specific version
            for version in model_info["versions"]:
                if version["id"] == version_id:
                    return version
            logger.error(f"Version {version_id} not found for model {model_name}")
            return None
        elif stage:
            # Get by stage
            if stage == "production":
                version_id = model_info.get("production_version")
            elif stage == "staging":
                version_id = model_info.get("staging_version")
            else:
                logger.error(f"Invalid stage: {stage}")
                return None
            
            if not version_id:
                logger.error(f"No {stage} version found for model {model_name}")
                return None
            
            # Find version by ID
            for version in model_info["versions"]:
                if version["id"] == version_id:
                    return version
            
            logger.error(f"Version {version_id} not found for model {model_name}")
            return None
        else:
            # Default to production, then staging, then latest
            version_id = model_info.get("production_version")
            if not version_id:
                version_id = model_info.get("staging_version")
            if not version_id and model_info["versions"]:
                # Get latest version
                return model_info["versions"][-1]
            
            if not version_id:
                logger.error(f"No versions found for model {model_name}")
                return None
            
            # Find version by ID
            for version in model_info["versions"]:
                if version["id"] == version_id:
                    return version
        
        return None
    
    def get_model_path(self, model_name, version_id=None, stage=None):
        """
        Get the path to a model version.
        
        Args:
            model_name (str): Name of the model
            version_id (str): Version ID (optional)
            stage (str): Stage to retrieve (production, staging) (optional)
            
        Returns:
            Path: Path to the model
        """
        version = self.get_model_version(model_name, version_id, stage)
        if not version:
            return None
        
        return self.registry_dir / version["id"] / "model"
    
    def promote_to_production(self, model_name, version_id):
        """
        Promote a model version to production.
        
        Args:
            model_name (str): Name of the model
            version_id (str): Version ID to promote
            
        Returns:
            bool: Success status
        """
        registry = self._load_registry()
        
        if model_name not in registry["models"]:
            logger.error(f"Model {model_name} not found in registry")
            return False
        
        # Verify version exists
        version_exists = False
        for version in registry["models"][model_name]["versions"]:
            if version["id"] == version_id:
                version_exists = True
                break
        
        if not version_exists:
            logger.error(f"Version {version_id} not found for model {model_name}")
            return False
        
        # Update production version
        registry["models"][model_name]["production_version"] = version_id
        
        # Save registry
        self._save_registry(registry)
        
        logger.info(f"Promoted model {model_name} version {version_id} to production")
        return True
    
    def list_models(self):
        """
        List all models in the registry.
        
        Returns:
            dict: Dictionary of models
        """
        registry = self._load_registry()
        return registry["models"]
    
    def list_versions(self, model_name):
        """
        List all versions of a model.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            list: List of version information
        """
        registry = self._load_registry()
        
        if model_name not in registry["models"]:
            logger.error(f"Model {model_name} not found in registry")
            return []
        
        return registry["models"][model_name]["versions"]
    
    def delete_version(self, model_name, version_id):
        """
        Delete a model version.
        
        Args:
            model_name (str): Name of the model
            version_id (str): Version ID to delete
            
        Returns:
            bool: Success status
        """
        registry = self._load_registry()
        
        if model_name not in registry["models"]:
            logger.error(f"Model {model_name} not found in registry")
            return False
        
        model_info = registry["models"][model_name]
        
        # Check if version is in use
        if model_info.get("production_version") == version_id:
            logger.error(f"Cannot delete production version {version_id}")
            return False
        
        if model_info.get("staging_version") == version_id:
            # Remove from staging
            model_info["staging_version"] = None
        
        # Remove from versions list
        model_info["versions"] = [v for v in model_info["versions"] if v["id"] != version_id]
        
        # Delete version directory
        version_dir = self.registry_dir / version_id
        if version_dir.exists():
            try:
                shutil.rmtree(version_dir)
            except Exception as e:
                logger.error(f"Error deleting version directory: {str(e)}")
        
        # Save registry
        self._save_registry(registry)
        
        logger.info(f"Deleted model {model_name} version {version_id}")
        return True
