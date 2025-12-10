# Configuration Reference

## Configuration System

The pipeline uses a hierarchical configuration system defined in [config.py](../config.py).

## Configuration Structure

```python
Config
├── experiment_name: str
├── output_dir: str
├── log_dir: str
├── model: ModelConfig
├── lora: LoRAConfig
├── data: DataConfig
├── training: TrainingConfig
└── eval: EvalConfig
```

## Quick Configuration

### Default Configuration

```python
from config import get_default_config

config = get_default_config()
# Full fine-tuning with default settings
```

### LoRA Configuration

```python
from config import get_lora_config

config = get_lora_config()
# LoRA training with default settings
```

### Custom Configuration

```python
from config import Config

config = Config()
config.experiment_name = "my_experiment"
config.training.learning_rate = 1e-4
config.save("my_config.json")
```

### Load Configuration

```python
config = Config.from_json("my_config.json")
```

## Configuration Sections

### ModelConfig

**Model architecture settings**

```python
@dataclass
class ModelConfig:
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    confidence_token: str = "<CONFIDENCE>"
    mlp_hidden_dims: List[int] = [1024, 256]
    mlp_dropout: float = 0.1
    mlp_activation: str = "gelu"
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
```

**Options**:
- `model_name`: HuggingFace model identifier
- `confidence_token`: Special token string
- `mlp_hidden_dims`: MLP layer dimensions (default: [1024, 256])
- `mlp_dropout`: Dropout rate (0.0-0.5, default: 0.1)
- `mlp_activation`: "gelu", "relu", "silu", or "tanh"
- `torch_dtype`: "float32", "float16", or "bfloat16"
- `device_map`: "auto", "cuda:0", or custom device map

**Examples**:
```python
# Larger MLP
config.model.mlp_hidden_dims = [2048, 1024, 512]

# Higher dropout for regularization
config.model.mlp_dropout = 0.2

# Use float16 instead of bfloat16
config.model.torch_dtype = "float16"
```

### LoRAConfig

**LoRA-specific settings**

```python
@dataclass
class LoRAConfig:
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = ["q_proj", "k_proj", "v_proj", "o_proj"]
```

**Options**:
- `use_lora`: Enable LoRA training
- `lora_r`: Rank of LoRA matrices (8-64, default: 16)
- `lora_alpha`: Scaling factor (typically 2× rank, default: 32)
- `lora_dropout`: LoRA-specific dropout (0.0-0.3, default: 0.1)
- `lora_target_modules`: Which layers to apply LoRA to

**Examples**:
```python
# Higher capacity LoRA
config.lora.lora_r = 32
config.lora.lora_alpha = 64

# Apply to all linear layers
config.lora.lora_target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]
```

### DataConfig

**Dataset and data loading settings**

```python
@dataclass
class DataConfig:
    # Dataset paths
    prm800k_path: str = "/workspace/datasets/PRM800K"
    reveal_path: str = "/workspace/datasets/reveal"
    eqasc_path: str = "/workspace/datasets/eQASC"

    # Dataset weights
    prm800k_weight: float = 0.5
    reveal_weight: float = 0.25
    eqasc_weight: float = 0.25

    # Dataset selection
    use_prm800k: bool = True
    use_reveal: bool = True
    use_eqasc: bool = True

    # Data loading
    max_train_samples: Optional[int] = None
    max_test_samples: Optional[int] = None
    max_length: int = 2048
    max_steps_per_example: int = 10
```

**Dataset Selection Examples**:
```python
# Math-focused training
config.data.use_prm800k = True
config.data.use_reveal = False
config.data.use_eqasc = False

# Multi-domain only
config.data.use_prm800k = False
config.data.use_reveal = True
config.data.use_eqasc = True
```

**Dataset Weighting Examples**:
```python
# Emphasize PRM800K
config.data.prm800k_weight = 0.7
config.data.reveal_weight = 0.15
config.data.eqasc_weight = 0.15

# Equal weighting
config.data.prm800k_weight = 0.33
config.data.reveal_weight = 0.33
config.data.eqasc_weight = 0.34
```

**Data Limits Examples**:
```python
# Limit for quick testing
config.data.max_train_samples = 1000
config.data.max_test_samples = 200

# Shorter sequences for memory savings
config.data.max_length = 1024
config.data.max_steps_per_example = 5
```

### TrainingConfig

**Training hyperparameters**

```python
@dataclass
class TrainingConfig:
    # Optimization
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # Learning rate schedule
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 500
    warmup_ratio: float = 0.0

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
```

**Learning Rate Examples**:
```python
# Conservative (full fine-tuning)
config.training.learning_rate = 1e-5

# Aggressive (LoRA)
config.training.learning_rate = 5e-4

# Use warmup ratio instead of steps
config.training.warmup_ratio = 0.05  # 5% of total steps
```

**Batch Size Examples**:
```python
# Large GPU
config.training.per_device_train_batch_size = 8
config.training.gradient_accumulation_steps = 2
# Effective batch size = 8 × 2 = 16

# Small GPU
config.training.per_device_train_batch_size = 2
config.training.gradient_accumulation_steps = 8
# Effective batch size = 2 × 8 = 16 (same effective)
```

**Schedule Examples**:
```python
# Linear schedule
config.training.lr_scheduler_type = "linear"

# Constant (no decay)
config.training.lr_scheduler_type = "constant"

# Cosine with longer warmup
config.training.lr_scheduler_type = "cosine"
config.training.warmup_steps = 1000
```

### EvalConfig

**Evaluation settings**

```python
@dataclass
class EvalConfig:
    compute_mse: bool = True
    compute_mae: bool = True
    compute_r2: bool = True
    compute_ece: bool = True
    ece_n_bins: int = 10
    eval_per_dataset: bool = True
    save_predictions: bool = True
    max_eval_batches: Optional[int] = 100  # None = evaluate all batches
```

**Examples**:
```python
# Disable expensive metrics
config.eval.compute_ece = False
config.eval.save_predictions = False

# More calibration bins
config.eval.ece_n_bins = 20

# Evaluate all batches (slower but more accurate)
config.eval.max_eval_batches = None

# Limit evaluation for faster feedback
config.eval.max_eval_batches = 50
```

## Command Line Overrides

Override config values via command line:

### train.py (Full Fine-Tuning)

```bash
python train.py \
    --config my_config.json \
    --experiment_name override_name \
    --max_steps 5000 \
    --learning_rate 1e-4 \
    --batch_size 2 \
    --logging_steps 10 \
    --eval_steps 100 \
    --save_steps 500 \
    --max_eval_batches 50
```

**Supported overrides for train.py**:
- `--config`: Path to config JSON file
- `--experiment_name`: Experiment name
- `--max_steps`: Total training steps
- `--learning_rate`: Learning rate
- `--batch_size`: Per-device batch size
- `--logging_steps`: Log metrics every N steps
- `--eval_steps`: Evaluate every N steps
- `--save_steps`: Save checkpoint every N steps
- `--max_eval_batches`: Max batches for evaluation (0 = all)

### train_lora.py (LoRA Fine-Tuning)

```bash
python train_lora.py \
    --config my_config.json \
    --experiment_name override_name \
    --max_steps 5000 \
    --learning_rate 1e-4 \
    --lora_r 16 \
    --lora_alpha 32
```

**Supported overrides for train_lora.py**:
- `--config`: Path to config JSON file
- `--experiment_name`: Experiment name
- `--max_steps`: Total training steps
- `--learning_rate`: Learning rate
- `--lora_r`: LoRA rank
- `--lora_alpha`: LoRA alpha scaling

## Complete Configuration Example

### Production Training Config

```python
from config import Config

config = Config()

# Experiment
config.experiment_name = "production_v1"
config.output_dir = "outputs"  # Checkpoints and logs go under outputs/{experiment_name}/

# Model
config.model.model_name = "Qwen/Qwen3-4B-Instruct-2507"
config.model.mlp_hidden_dims = [1024, 256]
config.model.mlp_dropout = 0.1

# LoRA
config.lora.use_lora = True
config.lora.lora_r = 16
config.lora.lora_alpha = 32

# Data
config.data.prm800k_weight = 0.5
config.data.reveal_weight = 0.25
config.data.eqasc_weight = 0.25

# Training
config.training.learning_rate = 1e-4
config.training.max_steps = 10000
config.training.warmup_steps = 500
config.training.per_device_train_batch_size = 4
config.training.gradient_accumulation_steps = 4

# Save
config.save("production_config.json")
```

### Testing Config

```python
config = Config()
config.experiment_name = "quick_test"
config.training.max_steps = 100
config.data.max_train_samples = 100
config.data.max_test_samples = 20
config.save("test_config.json")
```

## Utility Methods

### Get Effective Batch Size

```python
effective_batch = config.get_effective_batch_size()
# Returns: per_device_batch_size × gradient_accumulation_steps
```

### Get Dataset Configs

```python
# Get training dataset configs
train_configs = config.get_dataset_configs("train")

# Get test dataset configs
test_configs = config.get_dataset_configs("test")
```

### Convert to Dictionary

```python
config_dict = config.to_dict()
# Returns nested dictionary representation
```

## Best Practices

1. **Save configs**: Always save config with experiments for reproducibility
2. **Use presets**: Start with `get_default_config()` or `get_lora_config()`
3. **Version configs**: Name configs clearly (e.g., `production_v1.json`)
4. **Document changes**: Add comments explaining non-standard settings
5. **Test first**: Use test config with `max_steps=100` before full training

## See Also

- **[Training Guide](training.md)** - Training workflow
- **[Dataset Guide](datasets.md)** - Dataset configuration
- **[Best Practices](best-practices.md)** - Recommended settings
