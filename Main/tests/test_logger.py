"""Unit tests for logger utility."""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from utils.logger import Logger


class TestLogger:
    """Test Logger class functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)

    @pytest.fixture
    def logger(self, temp_dir):
        """Create logger instance for tests."""
        return Logger(
            log_dir=temp_dir,
            experiment_name="test_experiment"
        )

    def test_logger_initialization(self, logger, temp_dir):
        """Test logger initializes correctly."""
        assert logger.log_dir == temp_dir
        assert logger.experiment_name == "test_experiment"
        assert Path(temp_dir).exists()

    def test_log_message(self, logger):
        """Test logging messages."""
        logger.log("Test message", level="INFO")
        logger.close()
        
        # Check that log file was created
        log_files = list(Path(logger.log_dir).glob("*.log"))
        assert len(log_files) > 0

    def test_log_config(self, logger):
        """Test logging configuration."""
        config = {
            "model": "test_model",
            "learning_rate": 1e-4,
            "batch_size": 4
        }
        logger.log_config(config)
        logger.close()
        
        # Check config file exists
        config_file = Path(logger.log_dir) / "config.json"
        assert config_file.exists()
        
        # Check config content
        with open(config_file) as f:
            loaded_config = json.load(f)
        assert loaded_config["model"] == "test_model"

    def test_log_metrics(self, logger):
        """Test logging metrics."""
        metrics = {
            "loss": 0.5,
            "mse": 100.0,
            "mae": 8.5
        }
        logger.log_metrics(step=100, metrics=metrics, prefix="train/")
        logger.close()
        
        # Check metrics file exists
        metrics_files = list(Path(logger.log_dir).glob("*_metrics.jsonl"))
        assert len(metrics_files) > 0
        
        # Check metrics content
        with open(metrics_files[0]) as f:
            line = f.readline()
            data = json.loads(line)
            assert data["step"] == 100
            assert "train/loss" in data

    def test_log_epoch(self, logger):
        """Test logging epoch summary."""
        train_metrics = {"loss": 0.3, "mse": 50.0}
        val_metrics = {"loss": 0.4, "mse": 60.0}
        
        logger.log_epoch(epoch=1, train_metrics=train_metrics, val_metrics=val_metrics)
        logger.close()
        
        # Check that epoch was logged
        log_files = list(Path(logger.log_dir).glob("*.log"))
        assert len(log_files) > 0

    def test_save_results(self, logger):
        """Test saving final results."""
        results = {
            "best_val_loss": 0.3,
            "best_step": 1000,
            "final_metrics": {"mse": 50.0, "mae": 8.5}
        }
        logger.save_results(results)
        logger.close()
        
        # Check results file exists
        results_file = Path(logger.log_dir) / "final_results.json"
        assert results_file.exists()
        
        # Check results content
        with open(results_file) as f:
            loaded_results = json.load(f)
        assert loaded_results["best_val_loss"] == 0.3

    def test_load_metrics(self, logger):
        """Test loading metrics from file."""
        # Log some metrics first
        metrics = {"loss": 0.5, "mse": 100.0}
        logger.log_metrics(step=100, metrics=metrics, prefix="train/")
        logger.log_metrics(step=200, metrics=metrics, prefix="train/")
        logger.close()
        
        # Load metrics
        loaded = logger.load_metrics()
        assert len(loaded) >= 2

    def test_multiple_experiments(self, temp_dir):
        """Test multiple logger instances don't conflict."""
        logger1 = Logger(log_dir=temp_dir, experiment_name="exp1")
        logger2 = Logger(log_dir=temp_dir, experiment_name="exp2")
        
        logger1.log("Message 1")
        logger2.log("Message 2")
        
        logger1.close()
        logger2.close()
        
        # Both should have created log files
        log_files = list(Path(temp_dir).glob("*.log"))
        assert len(log_files) >= 2

