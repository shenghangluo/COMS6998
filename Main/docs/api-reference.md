# API Reference

## Overview

This document provides code-level documentation for the main modules. For usage examples, see the [Training Guide](training.md).

## Models

### ConfidenceMLP

**Location**: [models/mlp_head.py](../models/mlp_head.py)

**Purpose**: Standalone MLP head for confidence prediction

```python
from models.mlp_head import ConfidenceMLP

mlp = ConfidenceMLP(
    input_dim=4096,
    hidden_dims=[1024, 256],
    dropout=0.1,
    activation="gelu"
)

# Forward pass
embeddings = torch.randn(batch_size, num_steps, 4096)
predictions = mlp(embeddings)  # (batch_size, num_steps, 1)
```

**Parameters**:
- `input_dim` (int): Input embedding dimension
- `hidden_dims` (List[int]): Hidden layer dimensions
- `dropout` (float): Dropout probability
- `activation` (str): Activation function ("gelu", "relu", "silu", "tanh")

### ConfidenceModel

**Location**: [models/model.py](../models/model.py)

**Purpose**: Full fine-tuning model combining LLM backbone + MLP head

```python
from models.model import ConfidenceModel

model = ConfidenceModel(
    model_name="Qwen/Qwen3-4B-Instruct-2507",
    mlp_config={
        "input_dim": None,  # Auto-detected
        "hidden_dims": [1024, 256],
        "dropout": 0.1
    }
)

# Set confidence token ID
model.set_confidence_token_id(token_id)

# Forward pass
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    confidence_positions=confidence_positions,
    labels=labels
)

# Access outputs
predictions = outputs["predictions"]
loss = outputs["loss"]
```

**Methods**:
- `forward()`: Main forward pass
- `set_confidence_token_id()`: Set special token ID
- `generate_with_confidence()`: Generate text with confidence scores

### ConfidenceModelLoRA

**Location**: [models/lora_model.py](../models/lora_model.py)

**Purpose**: LoRA-based model for parameter-efficient fine-tuning

```python
from models.lora_model import ConfidenceModelLoRA

model = ConfidenceModelLoRA(
    model_name="Qwen/Qwen3-4B-Instruct-2507",
    lora_config={
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
    },
    mlp_config={
        "hidden_dims": [1024, 256]
    }
)
```

**Additional Methods**:
- `save_lora_adapter()`: Save LoRA weights
- `save_mlp_head()`: Save MLP head weights
- `merge_and_unload()`: Merge LoRA into base model

## Data

### ConfidenceDataset

**Location**: [data/dataset.py](../data/dataset.py)

**Purpose**: Unified PyTorch dataset for all three datasets

```python
from data.dataset import ConfidenceDataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({'additional_special_tokens': ['<CONFIDENCE>']})

dataset = ConfidenceDataset(
    tokenizer=tokenizer,
    dataset_configs=[
        {
            'name': 'prm800k',
            'path': '/workspace/datasets/PRM800K',
            'split': 'train',
            'weight': 0.5
        }
    ],
    confidence_token="<CONFIDENCE>",
    max_length=2048,
    max_steps_per_example=10
)

# Get item
item = dataset[0]
# Returns: {
#     'input_ids': Tensor,
#     'attention_mask': Tensor,
#     'confidence_positions': Tensor,
#     'labels': Tensor,
#     'source': str
# }
```

### Data Loaders

**Location**: [data/](../data/)

#### load_prm800k

```python
from data.prm800k_loader import load_prm800k

examples = load_prm800k(
    data_path="/workspace/datasets/PRM800K",
    split="train",  # or "test"
    phase=2,  # 1 or 2
    max_samples=None  # Limit samples
)

# Returns list of dicts:
# {
#     'problem': str,
#     'steps': List[str],
#     'confidences': List[float],
#     'ground_truth': str,
#     'source': 'prm800k'
# }
```

#### load_reveal

```python
from data.reveal_loader import load_reveal

examples = load_reveal(
    data_path="/workspace/datasets/reveal",
    split="eval",  # or "open"
    max_samples=None
)
```

#### load_eqasc

```python
from data.eqasc_loader import load_eqasc

examples = load_eqasc(
    data_path="/workspace/datasets/eQASC",
    split="train",  # or "dev", "test"
    max_samples=None
)
```

## Utilities

### Metrics

**Location**: [utils/metrics.py](../utils/metrics.py)

```python
from utils.metrics import compute_all_metrics

metrics = compute_all_metrics(
    predictions=pred_array,  # numpy array
    targets=label_array,     # numpy array
    n_bins=10,              # For ECE
    score_range=(0, 1)      # Score range
)

# Returns dict:
# {
#     'mse': float,
#     'mae': float,
#     'r2': float,
#     'ece': float,
#     'mae_low': float,      # MAE for [0, 0.25]
#     'mae_medium': float,   # MAE for (0.25, 0.75)
#     'mae_high': float,     # MAE for [0.75, 1.0]
#     'n_samples': int
# }
```

**Individual Metrics**:
```python
from utils.metrics import (
    compute_mse, compute_mae, compute_r2, compute_ece,
    compute_stratified_mae
)

mse = compute_mse(predictions, targets)
mae = compute_mae(predictions, targets)
r2 = compute_r2(predictions, targets)
ece = compute_ece(predictions, targets, n_bins=10, score_range=(0, 1))
stratified = compute_stratified_mae(predictions, targets)
```

### Logger

**Location**: [utils/logger.py](../utils/logger.py)

```python
from utils.logger import Logger

logger = Logger(
    log_dir="outputs/my_experiment/logs",
    experiment_name="my_experiment"
)

# Log message
logger.log("Training started", level="INFO")

# Log metrics
logger.log_metrics(
    step=100,
    metrics={'loss': 0.234, 'mse': 156.7},
    prefix="train/"
)

# Log configuration
logger.log_config(config.to_dict())

# Save results
logger.save_results({'best_loss': 0.15})

# Close logger
logger.close()
```

## Configuration

**Location**: [config.py](../config.py)

```python
from config import Config, get_default_config, get_lora_config

# Default config
config = get_default_config()

# LoRA config
config = get_lora_config()

# Custom config
config = Config()
config.experiment_name = "custom"
config.training.learning_rate = 1e-4

# Save/load
config.save("config.json")
config = Config.from_json("config.json")

# Utilities
effective_batch_size = config.get_effective_batch_size()
dataset_configs = config.get_dataset_configs("train")
config_dict = config.to_dict()
```

## Training Scripts

### train.py

**Location**: [train.py](../train.py)

**Usage**:
```bash
python train.py \
    --experiment_name my_experiment \
    --max_steps 10000 \
    --learning_rate 2e-5 \
    --batch_size 4
```

**Command Line Arguments**:
- `--config`: Path to config JSON
- `--experiment_name`: Experiment name
- `--max_steps`: Maximum training steps
- `--learning_rate`: Learning rate
- `--batch_size`: Per-device batch size

### train_lora.py

**Location**: [train_lora.py](../train_lora.py)

**Usage**:
```bash
python train_lora.py \
    --experiment_name my_lora_experiment \
    --max_steps 10000 \
    --learning_rate 1e-4 \
    --lora_r 16 \
    --lora_alpha 32
```

**Additional Arguments**:
- `--lora_r`: LoRA rank
- `--lora_alpha`: LoRA alpha

### evaluate.py

**Location**: [evaluate.py](../evaluate.py)

**Usage**:
```bash
python evaluate.py \
    --checkpoint /path/to/checkpoint \
    --output_dir /path/to/output \
    --batch_size 4 \
    --save_predictions
```

**Arguments**:
- `--checkpoint`: Checkpoint directory path
- `--config`: Config file (optional)
- `--output_dir`: Output directory
- `--batch_size`: Evaluation batch size
- `--save_predictions`: Save predictions to JSON

## Code Examples

### Custom Training Loop

```python
from transformers import AutoTokenizer
from models.lora_model import ConfidenceModelLoRA
from data.dataset import ConfidenceDataset
from config import Config

# Setup
config = Config()
tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
tokenizer.add_special_tokens({'additional_special_tokens': ['<CONFIDENCE>']})

# Model
model = ConfidenceModelLoRA(
    model_name=config.model.model_name,
    lora_config={'r': 16, 'lora_alpha': 32}
)
model.backbone.resize_token_embeddings(len(tokenizer))
model.set_confidence_token_id(tokenizer.convert_tokens_to_ids('<CONFIDENCE>'))

# Dataset
dataset = ConfidenceDataset(
    tokenizer=tokenizer,
    dataset_configs=config.get_dataset_configs("train")
)

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for batch in dataloader:
    outputs = model(**batch)
    loss = outputs['loss']
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### Custom Evaluation

```python
import numpy as np
from utils.metrics import compute_all_metrics

predictions = []
labels = []

model.eval()
with torch.no_grad():
    for batch in eval_dataloader:
        outputs = model(**batch)
        predictions.append(outputs['predictions'].cpu().numpy())
        labels.append(batch['labels'].cpu().numpy())

# Compute metrics
all_preds = np.concatenate(predictions)
all_labels = np.concatenate(labels)
metrics = compute_all_metrics(all_preds, all_labels)

print(f"MSE: {metrics['mse']:.2f}")
print(f"R²: {metrics['r2']:.4f}")
```

## See Also

- **[Architecture](architecture.md)** - Model architecture details
- **[Training Guide](training.md)** - Training procedures
- **[Dataset Guide](datasets.md)** - Dataset information
