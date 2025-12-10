# Quick Start Guide

Get training in 5 minutes with this streamlined guide.

## 1. Verify Setup

```bash
cd /workspace/Main
pip install -r requirements.txt
```

Test with a quick 100-step training run (~5 min):

```bash
python train_lora.py --experiment_name quick_test --max_steps 100
```

## 2. Start Training with LoRA

**Recommended for first-time users** - uses 50% less memory than full fine-tuning:

```bash
python train_lora.py \
    --experiment_name my_first_model \
    --max_steps 1000 \
    --learning_rate 1e-4
```

**Training time**: ~30-60 minutes on A100

**Output**:
- Checkpoints: `outputs/my_first_model/checkpoints/`
- Logs: `outputs/my_first_model/logs/`

## 3. Monitor Training

In another terminal:

```bash
tail -f outputs/my_first_model/logs/*.log
```

You should see:
- Training loss decreasing
- Validation metrics every 100 steps
- Checkpoints saved every 500 steps

## 4. Evaluate Results

```bash
python evaluate.py \
    --checkpoint outputs/my_first_model/checkpoints/checkpoint_step_1000 \
    --output_dir outputs/eval
```

**Expected metrics** (after 1000 steps):
- MSE: 0.02-0.04 (lower is better)
- MAE: 0.08-0.15 (overall mean absolute error)
- Stratified MAE: 0.05-0.15 (varies by confidence range)
- R²: 0.5-0.8 (goodness of fit)
- ECE: 0.02-0.10 (calibration error)

## Alternative: Full Fine-Tuning

For better performance (requires more memory):

```bash
python train.py \
    --experiment_name my_full_model \
    --max_steps 1000 \
    --learning_rate 2e-5
```

## Understanding the Output

### During Training

```
Step 100/10000 (1.0%)
train/loss: 0.0234
train/learning_rate: 9.5e-05
val/mse: 0.0199
val/mae: 0.0833
val/r2: 0.8208
```

- **loss**: MSE loss (should decrease)
- **mse/mae**: Validation metrics (lower is better)
- **r2**: Coefficient of determination (higher is better)

### Checkpoints

Saved at: `outputs/<experiment_name>/checkpoints/checkpoint_step_<N>/`

Each checkpoint contains:
- `backbone.pt` or `lora_adapter/` - Model weights
- `mlp_head.pt` - MLP head weights
- `config.json` - Configuration used

## What's Next?

### Improve Performance
- **Train longer**: Increase `--max_steps` to 10000
- **Adjust learning rate**: Try 5e-5 to 5e-4 for LoRA
- **Tune dataset weights**: See [Configuration Reference](configuration.md)

### Experiment
- **Different datasets**: Use only PRM800K or REVEAL (see [Dataset Guide](datasets.md))
- **MLP architecture**: Modify hidden dimensions in config
- **LoRA parameters**: Adjust rank and alpha with `--lora_r` and `--lora_alpha`

### Learn More
- **[Training Guide](training.md)** - Comprehensive training documentation
- **[Architecture](architecture.md)** - Understanding the model
- **[Best Practices](best-practices.md)** - Optimization strategies
- **[Troubleshooting](troubleshooting.md)** - Common issues

## Common Quick Issues

**Out of memory?**
```bash
# Use LoRA (recommended, uses less memory)
python train_lora.py --experiment_name my_exp --max_steps 1000

# Or reduce batch size for full fine-tuning
python train.py --batch_size 2 --experiment_name my_exp --max_steps 1000
```

**Training too slow?**
```bash
# Reduce max_steps for testing
python train_lora.py --experiment_name test --max_steps 100
```

**Want to see all options?**
```bash
python train_lora.py --help
python train.py --help
```

---

**Congratulations!** You've completed your first training run. Explore the [Training Guide](training.md) for advanced usage.
