"""
Unified YAML Configuration Loader for pytorch-3dunet integration
Supports train, test, and inference configurations with validation
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging

class ConfigLoader:
    """
    Unified configuration loader with validation and defaults
    
    Supports three configuration types:
    - train_config.yaml: Training configuration 
    - test_config.yaml: Testing/evaluation configuration
    - inference_config.yaml: Standalone inference configuration
    """
    
    def __init__(self, configs_dir: Optional[Union[str, Path]] = None):
        """
        Initialize ConfigLoader
        
        Args:
            configs_dir: Directory containing config files. If None, uses project default.
        """
        if configs_dir is None:
            # Default to project configs/ directory
            project_root = Path(__file__).parent.parent.parent
            self.configs_dir = project_root / "configs"
        else:
            self.configs_dir = Path(configs_dir)
        
        self.logger = logging.getLogger(__name__)
        
        if not self.configs_dir.exists():
            self.logger.warning(f"Configs directory not found: {self.configs_dir}")
    
    def load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load YAML configuration file with validation
        
        Args:
            config_path: Path to YAML config file
            
        Returns:
            Parsed configuration dictionary
        """
        config_path = Path(config_path)
        
        if not config_path.is_absolute():
            # If relative path, resolve relative to current working directory first
            if Path(config_path).exists():
                config_path = Path(config_path).resolve()
            else:
                # If not found, try configs directory
                config_path = self.configs_dir / config_path
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            self.logger.info(f"Loaded configuration from: {config_path}")
            return config
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file {config_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading config file {config_path}: {e}")
    
    def load_train_config(self, config_name: str = "train_config.yaml") -> Dict[str, Any]:
        """
        Load training configuration with defaults
        
        Args:
            config_name: Name of training config file
            
        Returns:
            Training configuration dictionary
        """
        config = self.load_config(config_name)
        
        # Validate required training sections
        required_sections = ['loaders', 'model', 'trainer', 'loss', 'optimizer']
        missing_sections = [section for section in required_sections if section not in config]
        
        if missing_sections:
            raise ValueError(f"Missing required sections in training config: {missing_sections}")
        
        # Apply training-specific defaults
        config = self._apply_training_defaults(config)
        
        return config
    
    def load_test_config(self, config_name: str = "test_config.yaml") -> Dict[str, Any]:
        """
        Load testing/evaluation configuration
        
        Args:
            config_name: Name of test config file
            
        Returns:
            Test configuration dictionary
        """
        config = self.load_config(config_name)
        
        # Validate required test sections
        required_sections = ['loaders', 'model', 'predictor']
        missing_sections = [section for section in required_sections if section not in config]
        
        if missing_sections:
            raise ValueError(f"Missing required sections in test config: {missing_sections}")
        
        return config
    
    def load_inference_config(self, config_name: str = "inference_config.yaml") -> Dict[str, Any]:
        """
        Load standalone inference configuration
        
        Args:
            config_name: Name of inference config file
            
        Returns:
            Inference configuration dictionary
        """
        config = self.load_config(config_name)
        
        # Validate required inference sections
        required_sections = ['model', 'inference']
        missing_sections = [section for section in required_sections if section not in config]
        
        if missing_sections:
            raise ValueError(f"Missing required sections in inference config: {missing_sections}")
        
        # Apply inference-specific defaults
        config = self._apply_inference_defaults(config)
        
        return config
    
    def _apply_training_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply default values for training configuration"""
        
        # Dataset defaults
        if 'dataset' not in config:
            config['dataset'] = {}
        
        dataset_defaults = {
            'sensor_size': [480, 640],
            'segment_duration_us': 20000,
            'num_bins': 8,
            'num_segments': 5
        }
        
        for key, default_value in dataset_defaults.items():
            if key not in config['dataset']:
                config['dataset'][key] = default_value
        
        # Model defaults (for UNet3D)
        model_defaults = {
            'in_channels': 1,
            'out_channels': 1,
            'final_sigmoid': False,  # For MSE loss, use False; for BCE-based losses, use True
            'f_maps': 32
        }
        
        for key, default_value in model_defaults.items():
            if key not in config['model']:
                config['model'][key] = default_value
        
        # Trainer defaults
        trainer_defaults = {
            'checkpoint_dir': 'checkpoints',
            'max_num_epochs': 100,
            'max_num_iterations': 1000000,
            'validate_after_iters': 1000,
            'log_after_iters': 100
        }
        
        for key, default_value in trainer_defaults.items():
            if key not in config['trainer']:
                config['trainer'][key] = default_value
        
        return config
    
    def _apply_inference_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply default values for inference configuration"""
        
        # Inference defaults
        inference_defaults = {
            'sensor_size': [480, 640],
            'segment_duration_us': 20000,
            'num_bins': 8,
            'num_segments': 5,
            'device': 'cuda' if self._has_cuda() else 'cpu'
        }
        
        for key, default_value in inference_defaults.items():
            if key not in config['inference']:
                config['inference'][key] = default_value
        
        return config
    
    def _has_cuda(self) -> bool:
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def validate_paths(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and convert relative paths to absolute paths
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configuration with validated paths
        """
        def convert_path(path_value):
            if isinstance(path_value, str):
                path = Path(path_value)
                if not path.is_absolute():
                    # Convert relative to absolute (relative to project root)
                    project_root = Path(__file__).parent.parent.parent
                    path = (project_root / path).resolve()
                return str(path)
            return path_value
        
        # Recursively process paths in config
        def process_config(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if 'path' in key.lower() or 'dir' in key.lower():
                        obj[key] = convert_path(value)
                    else:
                        obj[key] = process_config(value)
            elif isinstance(obj, list):
                return [process_config(item) for item in obj]
            return obj
        
        return process_config(config)
    
    def save_config(self, config: Dict[str, Any], output_path: Union[str, Path]) -> None:
        """
        Save configuration to YAML file
        
        Args:
            config: Configuration dictionary
            output_path: Output file path
        """
        output_path = Path(output_path)
        
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            
            self.logger.info(f"Configuration saved to: {output_path}")
            
        except Exception as e:
            raise RuntimeError(f"Error saving config to {output_path}: {e}")


# Convenience functions for direct usage
def load_train_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load training configuration"""
    loader = ConfigLoader()
    if config_path:
        return loader.load_config(config_path)
    return loader.load_train_config()

def load_test_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load test configuration"""
    loader = ConfigLoader()
    if config_path:
        return loader.load_config(config_path)
    return loader.load_test_config()

def load_inference_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load inference configuration"""
    loader = ConfigLoader()
    if config_path:
        return loader.load_config(config_path)
    return loader.load_inference_config()