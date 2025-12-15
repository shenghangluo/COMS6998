# Reproducibility Guide

This guide provides step-by-step instructions for reproducing the experimental results.

## Prerequisites

1. **Hardware**: GPU with 16GB+ VRAM (40GB+ recommended)
2. **Software**: Python 3.8+, CUDA 11.8+, required packages
3. **Datasets**: PRM800K, REVEAL, and eQASC datasets in correct locations

## Step 1: Environment Setup

### Install Dependencies

```bash
cd Main
pip install -r requirements.txt
```

### Verify Installation

```bash
# Check Python version
python --version  # Should be 3.8+

# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.version.cuda)"

# Check GPU
nvidia-smi
```

## Step 2: Dataset Preparation

### Verify Dataset Locations

```bash
# Check PRM800K
ls /workspace/datasets/PRM800K/phase2_train.jsonl
ls /workspace/datasets/PRM800K/phase2_test.jsonl

# Check REVEAL
ls /workspace/datasets/reveal/eval.jsonl
ls /workspace/datasets/reveal/open.jsonl

# Check eQASC
ls /workspace/datasets/eQASC/eqasc_train_grc.json
ls /workspace/datasets/eQASC/eqasc_test_grc.json
```

### Update Paths (if needed)

If datasets are in different locations, update paths in `config.py` or create a custom config.

## Step 3: Set Random Seed

The default configuration uses seed 42. To ensure reproducibility:

```python
from config import get_lora_config

config = get_lora_config()
config.training.seed = 42  # Default, but explicitly set for clarity
config.save("reproducibility_config.json")
```

## Step 4: Training

### Option A: LoRA Training (Recommended for Reproducibility)

```bash
python train_lora.py \
    --experiment_name reproducible_lora \
    --max_steps 10000 \
    --learning_rate 1e-4 \
    --config reproducibility_config.json
```

### Option B: Full Fine-Tuning

```bash
python train.py \
    --experiment_name reproducible_full \
    --max_steps 10000 \
    --learning_rate 2e-5 \
    --config reproducibility_config.json
```

### Expected Training Time

- **LoRA**: ~8-10 hours on A100 40GB
- **Full**: ~10-12 hours on A100 40GB

## Step 5: Monitor Training

### Check Logs

```bash
tail -f outputs/reproducible_lora/logs/*.log
```

### Expected Output Pattern

```
Step 100/10000 (1.0%)
train/loss: 0.0234
train/learning_rate: 9.5e-05
val/mse: 0.0199
val/mae: 0.0833
val/r2: 0.8208
```

## Step 6: Evaluation

### Evaluate Best Checkpoint

```bash
python evaluate.py \
    --checkpoint outputs/reproducible_lora/checkpoints/checkpoint_step_10000 \
    --output_dir outputs/eval_reproducible \
    --save_predictions
```

### Expected Metrics Range

After 10K steps, you should see:

- **MSE**: 0.015-0.025
- **MAE**: 0.08-0.12
- **R²**: 0.60-0.75
- **ECE**: 0.05-0.10

## Step 7: Verify Reproducibility

### Compare Results

```python
import json

# Load your results
with open('outputs/eval_reproducible/eval_metrics.json') as f:
    your_results = json.load(f)

# Compare with expected results
# (You should have baseline results for comparison)
print(f"MSE: {your_results['overall']['mse']:.4f}")
print(f"MAE: {your_results['overall']['mae']:.4f}")
print(f"R²: {your_results['overall']['r2']:.4f}")
```

## Reproducibility Checklist

- [ ] Python version matches (3.8+)
- [ ] CUDA version compatible
- [ ] All dependencies installed from requirements.txt
- [ ] Random seed set to 42
- [ ] Datasets in correct locations
- [ ] Config matches experimental setup
- [ ] GPU available and detected
- [ ] Training completed without errors
- [ ] Evaluation metrics within expected ranges

## Common Issues

### Different Results

If results differ significantly:

1. **Check random seed**: Ensure seed is set before any random operations
2. **Verify dataset versions**: Ensure using same dataset versions
3. **Check model version**: Ensure using same base model
4. **Verify hyperparameters**: Compare config files
5. **Check GPU**: Different GPUs may have slight numerical differences

### Non-Deterministic Behavior

For full determinism (may be slower):

```python
import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

## Reproducing Specific Experiments

### Experiment 1: LoRA Baseline

```bash
python train_lora.py \
    --experiment_name exp1_lora_baseline \
    --max_steps 10000 \
    --learning_rate 1e-4
```

### Experiment 2: Full Fine-Tuning

```bash
python train.py \
    --experiment_name exp2_full_finetuning \
    --max_steps 10000 \
    --learning_rate 2e-5
```

### Experiment 3: Different Learning Rates

```bash
# Low LR
python train_lora.py --experiment_name exp3_lr_low --learning_rate 5e-5

# High LR
python train_lora.py --experiment_name exp3_lr_high --learning_rate 5e-4
```

## Saving Experiment Configurations

Always save your configuration for reproducibility:

```python
from config import get_lora_config

config = get_lora_config()
config.experiment_name = "my_experiment"
# ... modify config ...
config.save("my_experiment_config.json")
```

Then use it:

```bash
python train_lora.py --config my_experiment_config.json
```

## Version Control

For full reproducibility, track:

- Git commit hash of code
- Exact package versions (`pip freeze > requirements_exact.txt`)
- Dataset versions/checksums
- Hardware specifications
- Random seed used

## Notes

- Small variations (< 1%) in metrics are normal due to floating-point precision
- Different GPU models may produce slightly different results
- Full determinism may require `torch.use_deterministic_algorithms(True)` (slower)
