# Evaluation Guide

## Quick Evaluation

```bash
python evaluate.py \
    --checkpoint outputs/my_experiment/checkpoint_step_10000 \
    --output_dir outputs/eval \
    --save_predictions
```

## Command Options

- `--checkpoint`: Path to checkpoint directory (required)
- `--config`: Path to config file (optional, loads from checkpoint if not provided)
- `--output_dir`: Where to save evaluation results (default: `outputs/eval`)
- `--batch_size`: Batch size for evaluation (default: 4)
- `--save_predictions`: Save individual predictions to JSON

## Metrics

### Regression Metrics

**Mean Squared Error (MSE)**:
- Primary optimization metric
- Lower is better
- Typical range: 0.01-0.05 (after training)
- Interpretation: Average squared error in [0, 1] scale

**Mean Absolute Error (MAE)**:
- Average absolute error across all predictions
- More interpretable than MSE
- Typical range: 0.08-0.15 (after training)
- Interpretation: Average error in confidence points [0, 1]

**R² (Coefficient of Determination)**:
- Goodness of fit metric
- Range: -∞ to 1.0 (1.0 is perfect)
- Typical range: 0.3-0.7 (after training)
- Interpretation:
  - 1.0: Perfect predictions
  - 0.5: Explains 50% of variance
  - 0.0: No better than mean baseline
  - <0.0: Worse than mean baseline

### Calibration Metrics

**Expected Calibration Error (ECE)**:
- Measures how well predictions match actual confidence
- Lower is better
- Range: 0 to 1
- Typical range: 0.05-0.15 (after training)

**How ECE works**:
1. Bin predictions into ranges (e.g., 0-0.1, 0.1-0.2, ..., 0.9-1.0)
2. For each bin, compare average predicted vs actual confidence
3. Weighted average of bin errors

### Stratified Metrics

**Stratified MAE (by Confidence Range)**:
- **Critical for multi-modal distributions** - reveals performance gaps across different confidence levels
- Computes MAE separately for three ranges:
  - **mae_low**: [0, 0.25] - Low confidence predictions
  - **mae_medium**: (0.25, 0.75) - Medium confidence predictions
  - **mae_high**: [0.75, 1.0] - High confidence predictions
- **Why it matters**: Overall MAE can be misleading if model is excellent at extremes (0, 1.0) but poor at medium confidence (0.5)
- Typical ranges (after training):
  - mae_low: 0.05-0.10
  - mae_medium: 0.08-0.15
  - mae_high: 0.05-0.12

### Interpreting Metrics

**Good Performance** (after training):
- MSE: <0.04
- MAE: <0.12
- R²: >0.5
- ECE: <0.12
- Stratified MAE all <0.15

**Excellent/Production Performance**:
- MSE: <0.02
- MAE: <0.08
- R²: >0.7
- ECE: <0.07
- Stratified MAE all <0.10

## Output Format

### Console Output

```
============================================================
                    Evaluation Results
============================================================

Overall Metrics:
  loss: 0.0235
  mse: 0.0157
  mae: 0.0923
  r2: 0.6234
  ece: 0.0789
  mae_low: 0.0742
  mae_medium: 0.1056
  mae_high: 0.0834
  n_samples: 12543

Per-Dataset Metrics:

  PRM800K:
    mse: 0.0145
    mae: 0.0891
    r2: 0.6543
    ece: 0.0678
    mae_low: 0.0234
    mae_medium: 0.0987
    mae_high: 0.0823
    n_samples: 5234

  REVEAL:
    mse: 0.0178
    mae: 0.1012
    r2: 0.5821
    ece: 0.0934
    mae_low: 0.0834
    mae_medium: 0.1123
    mae_high: 0.0945
    n_samples: 3421

  EQASC:
    mse: 0.0162
    mae: 0.0945
    r2: 0.6012
    ece: 0.0756
    mae_low: 0.0876
    mae_medium: 0.1001
    mae_high: 0.0912
    n_samples: 3888
============================================================
```

### JSON Output

Saved to `<output_dir>/eval_metrics.json`:

```json
{
  "overall": {
    "loss": 0.2345,
    "mse": 156.78,
    "mae": 9.23,
    "rmse": 12.52,
    "r2": 0.6234,
    "ece": 7.89,
    "correlation": 0.7654,
    "n_samples": 12543
  },
  "per_dataset": {
    "prm800k": {
      "mse": 145.32,
      "mae": 8.91,
      "r2": 0.6543,
      "ece": 6.78,
      "n_samples": 5234
    },
    "reveal": { ... },
    "eqasc": { ... }
  }
}
```

### Predictions Output

If `--save_predictions` is used, saved to `<output_dir>/predictions.json`:

```json
[
  {
    "source": "prm800k",
    "predictions": [0.953, 0.887, 0.921],
    "labels": [1.0, 1.0, 1.0]
  },
  {
    "source": "reveal",
    "predictions": [0.452, 0.789],
    "labels": [0.5, 1.0]
  }
]
```

## Per-Dataset Analysis

Evaluating on each dataset separately helps identify:

- **Domain-specific performance**: Which domains the model handles well
- **Overfitting**: If performance differs significantly between datasets
- **Calibration issues**: If confidence is well-calibrated across domains

### Analyzing Results

**If PRM800K performance is best**:
- Model learned correctness prediction well
- May be overfitting to math domain
- Consider increasing REVEAL/eQASC weights

**If REVEAL performance is worst**:
- Smaller dataset size (fewer examples to learn from)
- More complex attribution task
- Consider increasing REVEAL weight or training longer

**If eQASC performance varies widely**:
- Scores have low variance (most in 15-22 range)
- Automatic scores vs human labels
- Less reliable signal for training

## Comparing Models

### Compare Checkpoints

```bash
# Evaluate multiple checkpoints
for step in 1000 5000 10000; do
    python evaluate.py \
        --checkpoint outputs/my_experiment/checkpoint_step_$step \
        --output_dir outputs/eval_step_$step
done
```

### Compare Training Methods

```bash
# Full fine-tuning
python evaluate.py \
    --checkpoint outputs/full_model/checkpoint_step_10000 \
    --output_dir outputs/eval_full

# LoRA
python evaluate.py \
    --checkpoint outputs/lora_model/checkpoint_step_10000 \
    --output_dir outputs/eval_lora
```

### Aggregate Results

```python
import json

# Load both results
with open('outputs/eval_full/eval_metrics.json') as f:
    full_metrics = json.load(f)

with open('outputs/eval_lora/eval_metrics.json') as f:
    lora_metrics = json.load(f)

# Compare
print(f"Full MSE: {full_metrics['overall']['mse']:.2f}")
print(f"LoRA MSE: {lora_metrics['overall']['mse']:.2f}")
```

## Error Analysis

### Load Predictions

```python
import json
import numpy as np

with open('outputs/eval/predictions.json') as f:
    preds = json.load(f)

# Analyze errors
errors = []
for item in preds:
    pred = np.array(item['predictions'])
    label = np.array(item['labels'])
    errors.extend(np.abs(pred - label))

# Error distribution
print(f"Mean error: {np.mean(errors):.2f}")
print(f"Median error: {np.median(errors):.2f}")
print(f"95th percentile error: {np.percentile(errors, 95):.2f}")
```

### Identify Problem Cases

```python
# Find worst predictions
for item in preds:
    pred = np.array(item['predictions'])
    label = np.array(item['labels'])
    error = np.abs(pred - label)

    if error.max() > 30:  # Large error
        print(f"Source: {item['source']}")
        print(f"Predictions: {pred}")
        print(f"Labels: {label}")
        print(f"Max error: {error.max():.2f}")
        print()
```

## Validation During Training

Evaluation runs automatically during training every `eval_steps` (default: 100):

```
[INFO] Running evaluation...
[INFO] Step 100: mse=250.45, mae=12.34, r2=0.42
```

### Monitoring Validation

Watch for:

**Improving**: Validation metrics getting better
```
Step 100: mse=250.45
Step 200: mse=230.12  ✓ Improving
Step 300: mse=210.78  ✓ Improving
```

**Overfitting**: Training improves but validation degrades
```
Step 1000: train_loss=0.15, val_loss=0.22
Step 2000: train_loss=0.10, val_loss=0.25  ⚠ Overfitting
```

**Plateauing**: No improvement
```
Step 3000: mse=150.23
Step 4000: mse=149.87  → Consider stopping
Step 5000: mse=150.45  → Not improving
```

## Custom Evaluation

### Evaluate Specific Dataset

Modify config before evaluation:

```python
from config import Config

config = Config()

# Evaluate only on PRM800K
config.data.use_prm800k = True
config.data.use_reveal = False
config.data.use_eqasc = False

config.save("eval_prm800k.json")
```

```bash
python evaluate.py \
    --checkpoint <path> \
    --config eval_prm800k.json \
    --output_dir outputs/eval_prm800k
```

### Custom Metrics

Modify [utils/metrics.py](../utils/metrics.py) to add custom metrics:

```python
def compute_custom_metric(predictions, targets):
    # Your custom metric
    return metric_value

# Add to compute_all_metrics()
```

## Evaluation Best Practices

1. **Always evaluate on held-out test set**: Don't evaluate on training data
2. **Per-dataset analysis**: Check performance across all datasets
3. **Multiple checkpoints**: Evaluate several checkpoints to find best
4. **Error analysis**: Examine worst cases to understand failures
5. **Calibration**: Check ECE to ensure confidence is well-calibrated

## Interpretation Guidelines

### MSE vs MAE

- **MSE** penalizes large errors heavily
- **MAE** treats all errors equally
- If MSE >> MAE²: Large outlier errors exist
- If MSE ≈ MAE²: Errors are well-distributed

### R² Interpretation

- **R² = 0.7**: Model explains 70% of variance (good)
- **R² = 0.5**: Model explains 50% of variance (acceptable)
- **R² = 0.3**: Model explains 30% of variance (weak)
- **R² < 0**: Model worse than predicting mean (bad)

### ECE Interpretation

- **ECE < 5**: Well-calibrated (excellent)
- **ECE 5-10**: Reasonably calibrated (good)
- **ECE 10-20**: Poorly calibrated (needs improvement)
- **ECE > 20**: Very poorly calibrated (bad)

## Next Steps

- **Improve performance**: See [Best Practices](best-practices.md)
- **Debug issues**: See [Troubleshooting](troubleshooting.md)
- **Compare configurations**: See [Training Guide](training.md)
