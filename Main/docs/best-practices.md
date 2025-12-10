# Best Practices

## Training Strategies

### Start with LoRA

**Why**: Faster iteration, less memory, easier experimentation

```bash
# Quick experiments
python train_lora.py --max_steps 1000 --experiment_name exp_001
python train_lora.py --max_steps 1000 --experiment_name exp_002 --learning_rate 5e-4
```

### Progressive Training

**Phase 1**: Test with limited data (1 hour)
```python
config.training.max_steps = 1000
config.data.max_train_samples = 5000
```

**Phase 2**: Medium training (4 hours)
```python
config.training.max_steps = 5000
config.data.max_train_samples = None  # Use all
```

**Phase 3**: Full training (8-12 hours)
```python
config.training.max_steps = 10000
```

### Hyperparameter Tuning

**Test learning rates systematically**:
```bash
for lr in 5e-5 1e-4 5e-4; do
    python train_lora.py \
        --experiment_name lr_${lr} \
        --learning_rate ${lr} \
        --max_steps 2000
done
```

**Compare results**:
```python
import json

results = {}
for lr in ['5e-5', '1e-4', '5e-4']:
    with open(f'outputs/eval_lr_{lr}/eval_metrics.json') as f:
        results[lr] = json.load(f)

# Find best
best_lr = min(results.keys(), key=lambda lr: results[lr]['overall']['mse'])
print(f"Best learning rate: {best_lr}")
```

## Model Architecture

### MLP Head Design

**Default** (recommended for most cases):
```python
config.model.mlp_hidden_dims = [1024, 256]
```

**Larger capacity** (if underfitting):
```python
config.model.mlp_hidden_dims = [2048, 1024, 512]
config.model.mlp_dropout = 0.15  # Increase dropout
```

**Smaller/faster** (if overfitting or resource-constrained):
```python
config.model.mlp_hidden_dims = [512, 128]
config.model.mlp_dropout = 0.05  # Decrease dropout
```

### LoRA Configuration

**Default** (good balance):
```python
config.lora.lora_r = 16
config.lora.lora_alpha = 32
```

**Higher capacity** (better performance, more memory):
```python
config.lora.lora_r = 32
config.lora.lora_alpha = 64
```

**Lower capacity** (faster, less memory):
```python
config.lora.lora_r = 8
config.lora.lora_alpha = 16
```

## Dataset Strategies

### Dataset Selection by Goal

**Maximum performance** (use all data):
```python
config.data.use_prm800k = True
config.data.use_reveal = True
config.data.use_eqasc = True
config.data.prm800k_weight = 0.5
config.data.reveal_weight = 0.25
config.data.eqasc_weight = 0.25
```

**Math-focused applications**:
```python
config.data.use_prm800k = True
config.data.use_reveal = False
config.data.use_eqasc = False
```

**Multi-domain reasoning**:
```python
config.data.use_prm800k = False
config.data.use_reveal = True
config.data.use_eqasc = True
config.data.reveal_weight = 0.5
config.data.eqasc_weight = 0.5
```

### Curriculum Learning

**Phase 1**: Easier problems (3000 steps)
```python
config.data.prm800k_phase = 1  # Simpler problems
```

**Phase 2**: Harder problems (7000 steps)
```python
config.data.prm800k_phase = 2  # Complex problems
```

## Training Hyperparameters

### Learning Rate Guidelines

**Full Fine-Tuning**:
- Conservative: `1e-5`
- Default: `2e-5`
- Aggressive: `5e-5`

**LoRA**:
- Conservative: `5e-5`
- Default: `1e-4`
- Aggressive: `5e-4`

**Finding optimal LR**:
1. Start with default
2. If loss plateaus quickly → increase by 2-5×
3. If loss diverges/NaN → decrease by 2-5×

### Batch Size Strategy

**Effective batch size = per_device_batch × gradient_accumulation**

Target effective batch size: **16-32**

**Large GPU (A100 80GB)**:
```python
config.training.per_device_train_batch_size = 8
config.training.gradient_accumulation_steps = 2
# Effective = 16
```

**Medium GPU (A100 40GB)**:
```python
config.training.per_device_train_batch_size = 4
config.training.gradient_accumulation_steps = 4
# Effective = 16
```

**Small GPU or OOM**:
```python
config.training.per_device_train_batch_size = 2
config.training.gradient_accumulation_steps = 8
# Effective = 16
```

### Warmup and Scheduling

**Default** (cosine with warmup):
```python
config.training.lr_scheduler_type = "cosine"
config.training.warmup_steps = 500
```

**Longer warmup** (for aggressive LR):
```python
config.training.warmup_ratio = 0.1  # 10% of total steps
```

**No warmup** (for conservative LR):
```python
config.training.warmup_steps = 0
```

## Regularization

### Dropout

**Default**:
```python
config.model.mlp_dropout = 0.1
config.lora.lora_dropout = 0.1
```

**If overfitting**:
```python
config.model.mlp_dropout = 0.2  # Increase
```

**If underfitting**:
```python
config.model.mlp_dropout = 0.05  # Decrease
```

### Weight Decay

**Default**:
```python
config.training.weight_decay = 0.01
```

**Stronger regularization**:
```python
config.training.weight_decay = 0.05
```

### Gradient Clipping

**Default**:
```python
config.training.max_grad_norm = 1.0
```

**If training unstable**:
```python
config.training.max_grad_norm = 0.5  # More aggressive
```

## Evaluation Best Practices

### During Training

**Frequent evaluation** (more data, slower):
```python
config.training.eval_steps = 50
```

**Infrequent evaluation** (less data, faster):
```python
config.training.eval_steps = 200
```

**Default** (good balance):
```python
config.training.eval_steps = 100
```

### After Training

1. **Evaluate multiple checkpoints**:
   ```bash
   for step in 5000 7500 10000; do
       python evaluate.py \
           --checkpoint outputs/exp/checkpoint_step_${step}
   done
   ```

2. **Compare per-dataset performance**
3. **Analyze error cases**

## Memory Optimization

### Sequence Length

**Default**:
```python
config.data.max_length = 2048
```

**Memory-constrained**:
```python
config.data.max_length = 1024  # 50% less memory
```

**Long sequences needed**:
```python
config.data.max_length = 4096  # 2× more memory
```

### Steps Per Example

**Default**:
```python
config.data.max_steps_per_example = 10
```

**Memory-constrained**:
```python
config.data.max_steps_per_example = 5
```

## Reproducibility

### Set Random Seed

```python
config.training.seed = 42
```

### Save Configuration

```python
# Always save config with experiments
config.save(f"configs/{config.experiment_name}.json")
```

### Log Everything

- Use descriptive experiment names
- Save training logs
- Save evaluation results
- Document hyperparameter choices

## Production Checklist

Before deploying a model:

- [ ] Trained for sufficient steps (≥10K)
- [ ] Evaluated on all test sets
- [ ] Per-dataset metrics acceptable
- [ ] Calibration (ECE) checked
- [ ] Error analysis performed
- [ ] Configuration saved
- [ ] Checkpoints backed up
- [ ] Results documented

## Common Pitfalls to Avoid

1. **Not saving configs**: Always save configuration for reproducibility
2. **Ignoring validation metrics**: Monitor overfitting
3. **Using wrong learning rate**: LoRA needs higher LR than full fine-tuning
4. **Insufficient training**: 1000 steps is often too few
5. **Not checking per-dataset performance**: Model may fail on specific domains
6. **Skipping error analysis**: Understanding failures helps improvement
7. **Not comparing to baselines**: Always have a reference point

## Recommended Workflows

### Quick Experiment (2 hours)

```bash
# 1. Create config
python -c "
from config import get_lora_config
config = get_lora_config()
config.experiment_name = 'quick_exp'
config.training.max_steps = 2000
config.save('quick_exp.json')
"

# 2. Train
python train_lora.py --config quick_exp.json

# 3. Evaluate
python evaluate.py \
    --checkpoint outputs/quick_exp/checkpoint_step_2000 \
    --output_dir outputs/eval_quick
```

### Production Training (12 hours)

```bash
# 1. Create config
python -c "
from config import get_lora_config
config = get_lora_config()
config.experiment_name = 'production_v1'
config.training.max_steps = 10000
config.training.learning_rate = 1e-4
config.save('production_v1.json')
"

# 2. Train
python train_lora.py --config production_v1.json

# 3. Evaluate all checkpoints
for step in 5000 7500 10000; do
    python evaluate.py \
        --checkpoint outputs/production_v1/checkpoint_step_${step} \
        --output_dir outputs/eval_production_step_${step}
done

# 4. Select best checkpoint based on metrics
```

### Hyperparameter Search

```bash
# Test learning rates
for lr in 5e-5 1e-4 2e-4 5e-4; do
    python train_lora.py \
        --experiment_name hparam_lr_${lr} \
        --learning_rate ${lr} \
        --max_steps 3000
done

# Test LoRA ranks
for r in 8 16 32; do
    python train_lora.py \
        --experiment_name hparam_r_${r} \
        --lora_r ${r} \
        --lora_alpha $((r * 2)) \
        --max_steps 3000
done

# Evaluate all and compare
```

## Performance Targets

### After 1000 Steps (Quick Test)

- MSE: 0.02-0.04
- MAE: 0.08-0.15
- R²: 0.5-0.7
- ECE: 0.02-0.08

### After 10000 Steps (Full Training)

- MSE: <0.02
- MAE: <0.10
- R²: >0.7
- ECE: <0.05

### Production Quality

- MSE: <0.015
- MAE: <0.08
- R²: >0.8
- ECE: <0.03

## See Also

- **[Training Guide](training.md)** - Training procedures
- **[Configuration Reference](configuration.md)** - All config options
- **[Troubleshooting](troubleshooting.md)** - Common issues
- **[Evaluation Guide](evaluation.md)** - Metrics and analysis
