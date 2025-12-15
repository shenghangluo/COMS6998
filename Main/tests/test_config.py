"""Unit tests for configuration system."""

import pytest
import json
import tempfile
from pathlib import Path
from config import Config, get_lora_config


class TestConfig:
    """Test Config class functionality."""

    def test_config_initialization(self):
        """Test config initializes with defaults."""
        config = Config()
        assert config.experiment_name is not None
        assert hasattr(config, 'model')
        assert hasattr(config, 'training')
        assert hasattr(config, 'data')

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = Config()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert 'experiment_name' in config_dict
        assert 'model' in config_dict
        assert 'training' in config_dict

    def test_config_save_load(self):
        """Test saving and loading config."""
        config = Config()
        config.experiment_name = "test_save_load"
        config.training.learning_rate = 5e-5
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            config.save(temp_path)
            
            # Load config
            loaded_config = Config()
            loaded_config.load(temp_path)
            
            assert loaded_config.experiment_name == "test_save_load"
            assert loaded_config.training.learning_rate == 5e-5
        finally:
            Path(temp_path).unlink()

    def test_get_lora_config(self):
        """Test get_lora_config function."""
        config = get_lora_config()
        
        assert config.experiment_name is not None
        assert hasattr(config, 'lora')
        assert config.lora.lora_r > 0

    def test_config_modification(self):
        """Test modifying config values."""
        config = Config()
        original_lr = config.training.learning_rate
        
        config.training.learning_rate = 1e-3
        assert config.training.learning_rate == 1e-3
        assert config.training.learning_rate != original_lr

    def test_get_dataset_configs(self):
        """Test getting dataset configurations."""
        config = Config()
        
        train_configs = config.get_dataset_configs("train")
        assert isinstance(train_configs, list)
        
        test_configs = config.get_dataset_configs("test")
        assert isinstance(test_configs, list)

    def test_config_validation(self):
        """Test config value validation."""
        config = Config()
        
        # Test valid values
        config.training.learning_rate = 1e-4
        assert config.training.learning_rate == 1e-4
        
        config.training.max_steps = 1000
        assert config.training.max_steps == 1000

    def test_config_nested_attributes(self):
        """Test accessing nested config attributes."""
        config = Config()
        
        # Access nested attributes
        assert hasattr(config.model, 'model_name')
        assert hasattr(config.training, 'learning_rate')
        assert hasattr(config.data, 'max_length')

