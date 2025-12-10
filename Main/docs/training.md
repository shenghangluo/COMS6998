# Training Guide

## Overview

This guide covers both training modes: **full fine-tuning** and **LoRA** (parameter-efficient fine-tuning).

## Quick Comparison

| Aspect | Full Fine-Tuning | LoRA |
|--------|------------------|------|
| **Memory** | ~40GB | ~20GB |
| **Speed** | Baseline | 1.3× faster |
| **Performance** | Best | 90-95% of full |
| **Parameters Trained** | 4B + 5M | ~40M + 5M |
| **Use Case** | Maximum performance | Faster iteration, less memory |

**Recommendation**: Start with LoRA for experimentation, use full fine-tuning for production.

## Training with LoRA

### Basic Command

```bash
python train_lora.py \
    --experiment_name my_lora_experiment \
    --max_steps 10000 \
    --learning_rate 1e-4
```

### Available Options

```bash
python train_lora.py \
    --experiment_name my_experiment \
    --max_steps 10000 \
    --learning_rate 1e-4 \
    --lora_r 16 \
    --lora_alpha 32
```

### Parameters (train_lora.py)

- `--config`: Path to config JSON file
- `--experiment_name`: Unique identifier for this training run
- `--max_steps`: Total training steps (default: 10000)
- `--learning_rate`: Learning rate (default: 1e-4 for LoRA)
- `--lora_r`: LoRA rank (default: 16, higher = more capacity)
- `--lora_alpha`: LoRA alpha scaling (default: 32)

### Using Configuration File

```python
from config import get_lora_config

config = get_lora_config()
config.experiment_name = "custom_lora"
config.training.max_steps = 5000
config.lora.lora_r = 32  # Increase capacity
config.save("lora_config.json")
```

```bash
python train_lora.py --config lora_config.json
```

## Training with Full Fine-Tuning

### Basic Command

```bash
python train.py \
    --experiment_name my_full_experiment \
    --max_steps 10000 \
    --learning_rate 2e-5
```

### Available Options

```bash
python train.py \
    --experiment_name my_experiment \
    --max_steps 10000 \
    --learning_rate 2e-5 \
    --batch_size 4 \
    --logging_steps 10 \
    --eval_steps 100 \
    --save_steps 500
```

### Parameters (train.py)

- `--config`: Path to config JSON file
- `--experiment_name`: Unique identifier for this training run
- `--max_steps`: Total training steps (default: 10000)
- `--learning_rate`: Learning rate (default: 2e-5)
- `--batch_size`: Per-device batch size (default: 4)
- `--logging_steps`: Log metrics every N steps (default: 10)
- `--eval_steps`: Evaluate every N steps (default: 100)
- `--save_steps`: Save checkpoint every N steps (default: 500)
- `--max_eval_batches`: Max batches for evaluation (default: 100, 0 = all)

**Note**: Learning rate for full fine-tuning (2e-5) is lower than LoRA (1e-4).

## Training Configuration

### Key Hyperparameters

**Learning Rate**:
- Full fine-tuning: `1e-5` to `5e-5` (recommended: `2e-5`)
- LoRA: `5e-5` to `5e-4` (recommended: `1e-4`)

**Batch Size**:
- Per-device: `2` to `8` (default: `4`)
- Gradient accumulation: `2` to `8` (default: `4`)
- Effective batch size: `per_device × accumulation = 16`

**Training Steps**:
- Quick test: `100-500` steps
- Experiment: `1000-3000` steps
- Production: `10000+` steps

**Evaluation**:
- Eval frequency: Every `100` steps
- Save frequency: Every `500` steps

### Advanced Configuration

```python
from config import Config

config = Config()

# Training
config.training.learning_rate = 2e-5
config.training.max_steps = 10000
config.training.warmup_steps = 500
config.training.lr_scheduler_type = "cosine"
config.training.max_grad_norm = 1.0

# Batch size
config.training.per_device_train_batch_size = 4
config.training.gradient_accumulation_steps = 4  # Effective = 16

# Mixed precision
config.training.bf16 = True  # Use bfloat16

# Dataset selection
config.data.use_prm800k = True
config.data.use_reveal = True
config.data.use_eqasc = True

# Dataset weights
config.data.prm800k_weight = 0.5
config.data.reveal_weight = 0.25
config.data.eqasc_weight = 0.25

config.save("my_config.json")
```

## Monitoring Training

### Real-time Logs

```bash
# Watch training progress
tail -f outputs/my_experiment/logs/*.log
```

**Expected output**:
```
Step 100/10000 (1.0%)
train/loss: 0.0234
train/learning_rate: 9.5e-05
val/mse: 0.0199
val/mae: 0.0833
val/r2: 0.8208
```

### Metrics Files

Training metrics are saved in JSONL format:

```bash
cat outputs/my_experiment/logs/*_metrics.jsonl
```

Each line contains:
```json
{
  "step": 100,
  "timestamp": 1700050800.123,
  "train/loss": 0.0234,
  "train/learning_rate": 9.5e-05
}
```

## Checkpoints

### Checkpoint Structure

**Location**: `outputs/<experiment_name>/checkpoints/checkpoint_step_<N>/`

**Full fine-tuning**:
```
checkpoint_step_1000/
├── backbone.pt          # Model weights (~8GB)
├── mlp_head.pt          # MLP head (~20MB)
├── optimizer.pt         # Optimizer state
├── scheduler.pt         # LR scheduler state
└── config.json          # Training config
```

**LoRA**:
```
checkpoint_step_1000/
├── lora_adapter/        # LoRA weights (~200MB)
│   ├── adapter_config.json
│   └── adapter_model.bin
├── mlp_head.pt          # MLP head (~20MB)
├── optimizer.pt
├── scheduler.pt
└── config.json
```

### Best Model Selection

The training script automatically saves the checkpoint with the lowest validation loss:

```
New best validation loss: 0.0199
Checkpoint saved to outputs/my_experiment/checkpoints/checkpoint_step_1000
```

Look for "New best validation loss" in logs to identify the best checkpoint.

### Checkpoint Cleanup

The training script automatically keeps only the latest 3 checkpoints to save disk space.

## Training Strategies

### Dataset-Specific Training

**Math-focused** (PRM800K emphasis):
```python
config.data.prm800k_weight = 0.8
config.data.reveal_weight = 0.1
config.data.eqasc_weight = 0.1
```

**Multi-domain** (REVEAL + eQASC emphasis):
```python
config.data.prm800k_weight = 0.2
config.data.reveal_weight = 0.4
config.data.eqasc_weight = 0.4
```

### Progressive Training

Start with one dataset, then add others:

**Phase 1** - Learn from PRM800K:
```python
config.data.use_prm800k = True
config.data.use_reveal = False
config.data.use_eqasc = False
config.training.max_steps = 5000
```

**Phase 2** - Add other datasets:
```python
# Load Phase 1 checkpoint, then:
config.data.use_reveal = True
config.data.use_eqasc = True
config.training.max_steps = 10000
```

## Hyperparameter Tuning

### Learning Rate

Test different learning rates:

```bash
# LoRA
python train_lora.py --experiment_name lr_5e5 --learning_rate 5e-5
python train_lora.py --experiment_name lr_1e4 --learning_rate 1e-4
python train_lora.py --experiment_name lr_5e4 --learning_rate 5e-4

# Full fine-tuning
python train.py --experiment_name lr_1e5 --learning_rate 1e-5
python train.py --experiment_name lr_2e5 --learning_rate 2e-5
python train.py --experiment_name lr_5e5 --learning_rate 5e-5
```

Monitor validation loss to find optimal rate.

### LoRA Rank

Higher rank = more capacity:

```bash
python train_lora.py --experiment_name r8 --lora_r 8    # Lower capacity, faster
python train_lora.py --experiment_name r16 --lora_r 16  # Default
python train_lora.py --experiment_name r32 --lora_r 32  # Higher capacity
```

**Rule of thumb**: r=16 is sufficient for most cases.

### Batch Size (Full Fine-Tuning)

Adjust for memory:

```bash
# Large GPU (A100 80GB)
python train.py --batch_size 8

# Medium GPU (A100 40GB)
python train.py --batch_size 4  # Default

# Small GPU (or OOM)
python train.py --batch_size 2
```

Compensate with gradient accumulation to maintain effective batch size.

## Common Training Patterns

### Quick Experiment (1 hour)

```bash
python train_lora.py \
    --experiment_name quick_test \
    --max_steps 1000
```

### Production Training (8-12 hours)

```bash
python train.py \
    --experiment_name production_v1 \
    --max_steps 10000 \
    --learning_rate 2e-5 \
    --batch_size 4
```

### Memory-Constrained Training

```bash
# Use LoRA (default uses less memory)
python train_lora.py \
    --experiment_name low_memory \
    --max_steps 10000
```

Or edit config for more memory savings:
```python
config.data.max_length = 1024  # Shorter sequences
config.data.max_steps_per_example = 5  # Fewer steps
```

## Output Files

After training, you'll have:

```
outputs/
└── my_experiment/
    ├── checkpoints/
    │   ├── checkpoint_step_500/
    │   ├── checkpoint_step_1000/
    │   └── checkpoint_step_10000/  (final)
    └── logs/
        ├── train_*.log
        ├── *_metrics.jsonl
        ├── config.json
        └── final_results.json
```

## Next Steps

- **Evaluation**: See [Evaluation Guide](evaluation.md)
- **Tuning**: See [Best Practices](best-practices.md)
- **Troubleshooting**: See [Troubleshooting](troubleshooting.md)
