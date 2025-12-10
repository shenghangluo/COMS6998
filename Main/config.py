"""Configuration for confidence prediction training."""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
import json


@dataclass
class ModelConfig:
    """Configuration for the model."""
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    confidence_token: str = "<CONFIDENCE>"
    mlp_hidden_dims: List[int] = field(default_factory=lambda: [1024, 256])
    mlp_dropout: float = 0.1
    mlp_activation: str = "gelu"
    torch_dtype: str = "bfloat16"  # "float32", "float16", or "bfloat16"
    device_map: str = "auto"  # Use "auto" for automatic device placement, or specific device like "cuda:0"


@dataclass
class LoRAConfig:
    """Configuration for LoRA training."""
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )


@dataclass
class DataConfig:
    """Configuration for datasets."""
    # Dataset paths
    prm800k_path: str = "/workspace/datasets/PRM800K"
    reveal_path: str = "/workspace/datasets/reveal"
    eqasc_path: str = "/workspace/datasets/eQASC"

    # Dataset weights for sampling
    prm800k_weight: float = 0.5
    reveal_weight: float = 0.25
    eqasc_weight: float = 0.25

    # Dataset splits
    use_prm800k: bool = True
    use_reveal: bool = True
    use_eqasc: bool = True

    prm800k_phase: int = 2  # 1 or 2
    prm800k_train_split: str = "train"
    prm800k_test_split: str = "test"

    reveal_train_split: str = "eval"
    reveal_test_split: str = "open"

    eqasc_train_split: str = "train"
    eqasc_test_split: str = "test"

    # Data loading
    max_train_samples: Optional[int] = None  # None = use all
    max_test_samples: Optional[int] = None
    max_length: int = 2048
    max_steps_per_example: int = 10


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Optimization
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # Learning rate schedule
    lr_scheduler_type: str = "cosine"  # "linear", "cosine", "constant"
    warmup_steps: int = 500
    warmup_ratio: float = 0.0  # If > 0, overrides warmup_steps

    # Training steps
    max_steps: int = 10000
    eval_steps: int = 100
    save_steps: int = 500
    logging_steps: int = 10

    # Batch sizes
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4

    # Mixed precision
    fp16: bool = False
    bf16: bool = True

    # Other
    seed: int = 42
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True


@dataclass
class EvalConfig:
    """Configuration for evaluation."""
    # Metrics
    compute_mse: bool = True
    compute_mae: bool = True
    compute_r2: bool = True
    compute_ece: bool = True
    ece_n_bins: int = 10

    # Per-dataset evaluation
    eval_per_dataset: bool = True

    # Save predictions
    save_predictions: bool = True

    # Max batches to evaluate (None = evaluate all)
    max_eval_batches: Optional[int] = 100  # Default to 100 batches for faster validation


@dataclass
class Config:
    """Main configuration combining all sub-configs."""
    # Experiment info
    experiment_name: str = "confidence_prediction"
    output_dir: str = "/workspace/Main/outputs"  # Base directory for all outputs
    log_dir: str = None  # Deprecated, logs now go in output_dir/{experiment_name}/logs
    checkpoint_dir: str = None  # Deprecated, checkpoints now go in output_dir/{experiment_name}/checkpoints

    # Sub-configs
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return asdict(self)

    def save(self, filepath: str):
        """Save config to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Config saved to {filepath}")

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'Config':
        """Load config from dictionary."""
        model_config = ModelConfig(**config_dict.get('model', {}))
        lora_config = LoRAConfig(**config_dict.get('lora', {}))
        data_config = DataConfig(**config_dict.get('data', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        eval_config = EvalConfig(**config_dict.get('eval', {}))

        return cls(
            experiment_name=config_dict.get('experiment_name', 'confidence_prediction'),
            output_dir=config_dict.get('output_dir', '/workspace/checkpoints'),
            log_dir=config_dict.get('log_dir', '/workspace/Main/logs'),
            model=model_config,
            lora=lora_config,
            data=data_config,
            training=training_config,
            eval=eval_config
        )

    @classmethod
    def from_json(cls, filepath: str) -> 'Config':
        """Load config from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def get_effective_batch_size(self) -> int:
        """Get effective batch size (per_device * accumulation * num_gpus)."""
        # For simplicity, assume single GPU here
        return (self.training.per_device_train_batch_size *
                self.training.gradient_accumulation_steps)

    def get_dataset_configs(self, split: str = "train") -> List[Dict]:
        """
        Get dataset configurations for the unified dataset.

        Args:
            split: "train" or "test"

        Returns:
            List of dataset config dicts
        """
        configs = []

        # PRM800K
        if self.data.use_prm800k:
            configs.append({
                'name': 'prm800k',
                'path': self.data.prm800k_path,
                'split': self.data.prm800k_train_split if split == "train" else self.data.prm800k_test_split,
                'phase': self.data.prm800k_phase,
                'weight': self.data.prm800k_weight,
                'max_samples': self.data.max_train_samples if split == "train" else self.data.max_test_samples
            })

        # REVEAL
        if self.data.use_reveal:
            configs.append({
                'name': 'reveal',
                'path': self.data.reveal_path,
                'split': self.data.reveal_train_split if split == "train" else self.data.reveal_test_split,
                'weight': self.data.reveal_weight,
                'max_samples': self.data.max_train_samples if split == "train" else self.data.max_test_samples
            })

        # eQASC
        if self.data.use_eqasc:
            configs.append({
                'name': 'eqasc',
                'path': self.data.eqasc_path,
                'split': self.data.eqasc_train_split if split == "train" else self.data.eqasc_test_split,
                'weight': self.data.eqasc_weight,
                'max_samples': self.data.max_train_samples if split == "train" else self.data.max_test_samples
            })

        return configs


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()


def get_lora_config() -> Config:
    """Get configuration for LoRA training."""
    config = Config()
    config.experiment_name = "confidence_prediction_lora"
    config.lora.use_lora = True
    config.training.learning_rate = 1e-4  # Higher LR for LoRA
    return config


if __name__ == "__main__":
    # Test configurations
    print("Testing configurations...")

    # Default config
    config = get_default_config()
    print("\nDefault Config:")
    print(f"  Model: {config.model.model_name}")
    print(f"  Use LoRA: {config.lora.use_lora}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Effective batch size: {config.get_effective_batch_size()}")

    # Save and load
    config.save("/workspace/logs/test_config.json")

    loaded_config = Config.from_json("/workspace/logs/test_config.json")
    print("\nLoaded config successfully!")

    # Get dataset configs
    train_configs = config.get_dataset_configs("train")
    print(f"\nTrain dataset configs: {len(train_configs)} datasets")
    for dc in train_configs:
        print(f"  - {dc['name']}: weight={dc['weight']}, split={dc['split']}")

    # LoRA config
    lora_config = get_lora_config()
    print(f"\nLoRA Config:")
    print(f"  Use LoRA: {lora_config.lora.use_lora}")
    print(f"  LoRA r: {lora_config.lora.lora_r}")
    print(f"  Learning rate: {lora_config.training.learning_rate}")
