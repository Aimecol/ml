"""Configuration management utilities."""

import yaml
import os
from typing import Dict, Any


class ConfigManager:
    """Manages YAML-based project configuration."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to config.yaml file. If None, uses default location.
        """
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__),
                '../../config/config.yaml'
            )
        
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config if config is not None else {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated key.
        
        Args:
            key: Configuration key (e.g., 'data.raw_path')
            default: Default value if key not found
        
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def get_all(self) -> Dict[str, Any]:
        """Get entire configuration."""
        return self.config
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by dot-separated key.
        
        Args:
            key: Configuration key
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, output_path: str = None) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            output_path: Path to save config. If None, uses original path.
        """
        output_path = output_path or self.config_path
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)


def load_config(config_path: str = None) -> ConfigManager:
    """Load and return configuration manager."""
    return ConfigManager(config_path)
