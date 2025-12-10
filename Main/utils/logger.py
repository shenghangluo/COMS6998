"""Simple file-based logger for training."""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class Logger:
    """
    Simple logger that writes training metrics to a file.

    Args:
        log_dir: Directory to save logs
        experiment_name: Name of the experiment
        log_to_console: Whether to also print to console (default: True)
    """

    def __init__(
        self,
        log_dir: str = "/workspace/Main/logs",
        experiment_name: str = "confidence_training",
        log_to_console: bool = True
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_name = experiment_name
        self.log_to_console = log_to_console

        # Create log files
        self.log_file = self.log_dir / "train.log"
        self.metrics_file = self.log_dir / "metrics.jsonl"

        # Initialize log files
        self._write_to_file(self.log_file, f"=== Training Log: {experiment_name} ===\n")
        self._write_to_file(self.log_file, f"Started at: {datetime.now()}\n\n")

        print(f"Logger initialized. Logs will be saved to:")
        print(f"  {self.log_file}")
        print(f"  {self.metrics_file}")

    def log(self, message: str, level: str = "INFO"):
        """
        Log a message.

        Args:
            message: Message to log
            level: Log level (INFO, WARNING, ERROR, DEBUG)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp}] [{level}] {message}"

        # Write to file
        self._write_to_file(self.log_file, formatted_message + "\n")

        # Print to console if enabled
        if self.log_to_console:
            print(formatted_message)

    def log_metrics(
        self,
        step: int,
        metrics: Dict[str, Any],
        prefix: str = ""
    ):
        """
        Log metrics for a training step.

        Args:
            step: Training step number
            metrics: Dict of metric name -> value
            prefix: Optional prefix for metric names (e.g., "train/", "val/")
        """
        timestamp = time.time()

        # Create metrics entry
        entry = {
            "step": step,
            "timestamp": timestamp,
            "datetime": datetime.now().isoformat(),
        }

        # Add prefixed metrics
        for key, value in metrics.items():
            metric_name = f"{prefix}{key}" if prefix else key
            entry[metric_name] = value

        # Write to metrics file (JSONL format)
        self._write_to_file(self.metrics_file, json.dumps(entry) + "\n")

        # Also log to text file
        metrics_str = ", ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                  for k, v in metrics.items()])
        self.log(f"Step {step}: {metrics_str}")

    def log_config(self, config: Dict[str, Any]):
        """
        Log training configuration.

        Args:
            config: Configuration dict
        """
        self.log("=== Training Configuration ===")
        for key, value in config.items():
            self.log(f"  {key}: {value}")
        self.log("=" * 50)

        # Also save as JSON
        config_file = self.log_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

    def log_epoch(self, epoch: int, train_metrics: Dict, val_metrics: Optional[Dict] = None):
        """
        Log end-of-epoch summary.

        Args:
            epoch: Epoch number
            train_metrics: Training metrics
            val_metrics: Validation metrics (optional)
        """
        self.log(f"\n{'='*50}")
        self.log(f"Epoch {epoch} Summary:")
        self.log(f"{'='*50}")

        self.log("Training Metrics:")
        for key, value in train_metrics.items():
            if isinstance(value, float):
                self.log(f"  {key}: {value:.4f}")
            else:
                self.log(f"  {key}: {value}")

        if val_metrics:
            self.log("Validation Metrics:")
            for key, value in val_metrics.items():
                if isinstance(value, float):
                    self.log(f"  {key}: {value:.4f}")
                else:
                    self.log(f"  {key}: {value}")

        self.log(f"{'='*50}\n")

    def save_checkpoint_info(self, step: int, checkpoint_path: str, metrics: Dict):
        """
        Log checkpoint save information.

        Args:
            step: Training step
            checkpoint_path: Path where checkpoint was saved
            metrics: Metrics at checkpoint time
        """
        self.log(f"Checkpoint saved at step {step}")
        self.log(f"  Path: {checkpoint_path}")
        self.log(f"  Metrics: {metrics}")

    def save_results(self, results: Dict[str, Any], filename: str = "final_results.json"):
        """
        Save final results to JSON file.

        Args:
            results: Results dict to save
            filename: Name of output file
        """
        results_file = self.log_dir / filename
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        self.log(f"Results saved to {results_file}")

    def close(self):
        """Close the logger and write footer."""
        self._write_to_file(self.log_file, f"\nTraining completed at: {datetime.now()}\n")
        self.log("Logger closed.")

    def _write_to_file(self, filepath: Path, content: str):
        """Write content to file."""
        with open(filepath, 'a') as f:
            f.write(content)

    def load_metrics(self) -> list:
        """
        Load all logged metrics from the metrics file.

        Returns:
            List of metric dicts
        """
        metrics = []
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                for line in f:
                    metrics.append(json.loads(line))
        return metrics


if __name__ == "__main__":
    # Test logger
    print("Testing Logger...")

    logger = Logger(
        log_dir="/workspace/Main/logs",
        experiment_name="test_experiment"
    )

    # Log config
    config = {
        "model": "Qwen3-4B",
        "learning_rate": 2e-5,
        "batch_size": 4,
        "max_steps": 10000
    }
    logger.log_config(config)

    # Log some training steps
    for step in range(5):
        metrics = {
            "loss": 0.5 - step * 0.05,
            "mse": 100 - step * 10,
            "mae": 8 - step * 0.5
        }
        logger.log_metrics(step, metrics, prefix="train/")

    # Log validation metrics
    val_metrics = {
        "loss": 0.3,
        "mse": 75,
        "mae": 6.5,
        "r2": 0.85
    }
    logger.log_metrics(5, val_metrics, prefix="val/")

    # Log epoch summary
    logger.log_epoch(
        epoch=1,
        train_metrics={"loss": 0.25, "mse": 50},
        val_metrics={"loss": 0.3, "mse": 75}
    )

    # Save final results
    final_results = {
        "best_val_loss": 0.3,
        "best_step": 5,
        "final_metrics": val_metrics
    }
    logger.save_results(final_results)

    logger.close()

    print("\nLogger test completed!")
