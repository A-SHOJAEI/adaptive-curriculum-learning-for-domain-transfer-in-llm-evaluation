"""Configuration management utilities."""

import logging
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml


class Config:
    """Configuration manager for the adaptive curriculum learning framework.

    This class handles loading, validation, and access to configuration parameters
    from YAML files. It also sets up reproducibility by managing random seeds.
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        """Initialize configuration manager.

        Args:
            config_path: Path to YAML configuration file. If None, uses default config.
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent.parent / "configs" / "default.yaml"

        self.config_path = Path(config_path)
        self._config = self._load_config()
        self._setup_logging()
        self._set_seeds()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file.

        Returns:
            Dictionary containing configuration parameters.

        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            yaml.YAMLError: If configuration file is malformed.
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logging.info(f"Loaded configuration from {self.config_path}")
            return config
        except FileNotFoundError:
            logging.error(f"Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logging.error(f"Error parsing configuration file: {e}")
            raise

    def _setup_logging(self) -> None:
        """Set up logging configuration."""
        log_config = self.get('logging', {})
        level = getattr(logging, log_config.get('level', 'INFO').upper())
        format_str = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        handlers = [logging.StreamHandler()]

        if log_config.get('save_to_file', False):
            log_file = log_config.get('log_file', 'training.log')
            handlers.append(logging.FileHandler(log_file))

        logging.basicConfig(
            level=level,
            format=format_str,
            handlers=handlers,
            force=True
        )

    def _set_seeds(self) -> None:
        """Set random seeds for reproducibility."""
        seed = self.get('seed', 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logging.info(f"Set random seeds to {seed}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.

        Args:
            key: Configuration key (supports nested keys with dots, e.g., 'model.name').
            default: Default value if key is not found.

        Returns:
            Configuration value or default.
        """
        keys = key.split('.')
        value = self._config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any) -> None:
        """Set configuration value by key.

        Args:
            key: Configuration key (supports nested keys with dots).
            value: Value to set.
        """
        keys = key.split('.')
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values.

        Args:
            updates: Dictionary of configuration updates.
        """
        def _deep_update(base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
            for key, value in update_dict.items():
                if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                    _deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value

        _deep_update(self._config, updates)

    def save(self, path: Optional[str] = None) -> None:
        """Save current configuration to YAML file.

        Args:
            path: Output path. If None, overwrites original config file.
        """
        output_path = Path(path) if path else self.config_path

        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, default_flow_style=False, indent=2)

        logging.info(f"Saved configuration to {output_path}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Deep copy of configuration dictionary.
        """
        import copy
        return copy.deepcopy(self._config)

    def __getitem__(self, key: str) -> Any:
        """Dict-like access to configuration values."""
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Dict-like setting of configuration values."""
        self.set(key, value)

    def __contains__(self, key: str) -> bool:
        """Check if configuration key exists."""
        return self.get(key) is not None