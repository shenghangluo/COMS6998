# Experimental Setup

This document provides detailed information about the experimental setup used for training and evaluating confidence prediction models.

## Hardware Specifications

### Primary Training Environment

- **GPU**: NVIDIA A100 40GB (or equivalent)
- **CPU**: Multi-core processor (8+ cores recommended)
- **RAM**: 32GB+ recommended
- **Storage**: 50GB+ free space for datasets and checkpoints

### Minimum Requirements

- **GPU**: NVIDIA GPU with 16GB+ VRAM (for LoRA training)
- **CPU**: 4+ cores
- **RAM**: 16GB minimum
- **Storage**: 30GB+ free space

## Software Environment

### Operating System

- **OS**: Linux (Ubuntu 20.04+ recommended) or compatible
- **Shell**: Bash or compatible shell

### Python Environment

- **Python Version**: 3.8 or higher (tested with 3.8, 3.9, 3.10)
- **Package Manager**: pip

### Key Dependencies

See `requirements.txt` for complete list. Core dependencies:

```
torch>=2.0.0
transformers>=4.40.0
accelerate>=0.20.0
peft>=0.7.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
```

### CUDA

- **CUDA Version**: 11.8+ (compatible with PyTorch)
- **cuDNN**: Version compatible with CUDA version

### Verification Commands

```bash
# Check Python version
python --version

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"

# Check GPU
nvidia-smi
```

## Dataset Versions and Sources

### PRM800K

- **Source**: Process Reward Model 800K dataset
- **Path**: `/workspace/datasets/PRM800K/`
- **Files Used**:
  - `phase2_train.jsonl` (training)
  - `phase2_test.jsonl` (testing)
- **Size**: ~97K examples
- **Format**: JSONL with reasoning steps and correctness ratings

### REVEAL

- **Source**: REVEAL attribution dataset
- **Path**: `/workspace/datasets/reveal/`
- **Files Used**:
  - `eval.jsonl` (training)
  - `open.jsonl` (testing)
- **Size**: ~6K examples
- **Format**: JSONL with questions, steps, and evidence attribution

### eQASC

- **Source**: eQASC science QA dataset
- **Path**: `/workspace/datasets/eQASC/`
- **Files Used**:
  - `eqasc_train_grc.json` (training)
  - `eqasc_test_grc.json` (testing)
- **Size**: ~119K reasoning chains
- **Format**: JSON with questions, reasoning chains, and relevance scores

## Model Configuration

### Base Model

- **Model**: Qwen/Qwen3-4B-Instruct-2507
- **Parameters**: ~4B parameters
- **Tokenizer**: Qwen tokenizer with special `<CONFIDENCE>` token added

### Training Configurations

#### LoRA Configuration (Default)

```python
lora_r = 16
lora_alpha = 32
lora_dropout = 0.05
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
```

#### Full Fine-Tuning Configuration

- All model parameters are trainable
- Uses gradient checkpointing for memory efficiency

#### MLP Head Configuration

```python
hidden_dims = [1024, 256]
dropout = 0.1
activation = "gelu"
output_activation = "sigmoid"  # Constrains output to [0, 1]
```

## Hyperparameter Settings

### Default Training Hyperparameters

#### LoRA Training

- **Learning Rate**: 1e-4
- **Batch Size**: 4 (per device)
- **Gradient Accumulation**: 4 steps
- **Effective Batch Size**: 16
- **Max Steps**: 10,000
- **Warmup Steps**: 500
- **Learning Rate Schedule**: Cosine
- **Weight Decay**: 0.01
- **Max Gradient Norm**: 1.0
- **Mixed Precision**: bfloat16

#### Full Fine-Tuning

- **Learning Rate**: 2e-5
- **Batch Size**: 4 (per device)
- **Gradient Accumulation**: 4 steps
- **Effective Batch Size**: 16
- **Max Steps**: 10,000
- **Warmup Steps**: 500
- **Learning Rate Schedule**: Cosine
- **Weight Decay**: 0.01
- **Max Gradient Norm**: 1.0
- **Mixed Precision**: bfloat16

### Data Configuration

- **Max Sequence Length**: 2048 tokens
- **Max Steps Per Example**: 10 steps
- **Dataset Weights**:
  - PRM800K: 0.5
  - REVEAL: 0.25
  - eQASC: 0.25

## Reproducibility

### Random Seeds

All experiments use a fixed random seed for reproducibility:

- **Seed**: 42 (default, configurable via `config.training.seed`)

### Seed Setting

The seed is set for:

- PyTorch random number generator
- NumPy random number generator
- Python's random module
- CUDA random number generator (if available)

### Deterministic Operations

For full reproducibility, you may need to set:

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

**Note**: This may reduce training speed.

## Evaluation Protocol

### Evaluation Metrics

- **MSE** (Mean Squared Error): Primary optimization metric
- **MAE** (Mean Absolute Error): Interpretable error metric
- **R²** (Coefficient of Determination): Goodness of fit
- **ECE** (Expected Calibration Error): Calibration quality
- **Stratified MAE**: MAE by confidence range (low/medium/high)

### Evaluation Frequency

- **During Training**: Every 100 steps (configurable)
- **Final Evaluation**: After training completion
- **Checkpoint Evaluation**: On best validation loss checkpoints

### Test Sets

- **PRM800K**: `phase2_test.jsonl`
- **REVEAL**: `open.jsonl`
- **eQASC**: `eqasc_test_grc.json`

## Experiment Tracking

### Logging

- **Log Directory**: `outputs/{experiment_name}/logs/`
- **Metrics Format**: JSONL (one line per step)
- **Config Saving**: JSON format saved with each experiment
- **Checkpoint Saving**: Every 500 steps (configurable)

### Checkpoint Structure

```
outputs/{experiment_name}/checkpoints/checkpoint_step_{N}/
├── config.json          # Training configuration
├── lora_adapter/        # LoRA weights (LoRA training)
│   ├── adapter_config.json
│   └── adapter_model.bin
├── mlp_head.pt          # MLP head weights
├── optimizer.pt         # Optimizer state
└── scheduler.pt         # LR scheduler state
```

## Version Information

### Software Versions Used

- **PyTorch**: 2.0.0+
- **Transformers**: 4.40.0+
- **PEFT**: 0.7.0+
- **Accelerate**: 0.20.0+

### Model Version

- **Base Model**: Qwen/Qwen3-4B-Instruct-2507 (as of experiment date)

## Running Experiments

### Standard Training Command

```bash
# LoRA training
python train_lora.py \
    --experiment_name my_experiment \
    --max_steps 10000 \
    --learning_rate 1e-4

# Full fine-tuning
python train.py \
    --experiment_name my_experiment \
    --max_steps 10000 \
    --learning_rate 2e-5
```

### Evaluation Command

```bash
python evaluate.py \
    --checkpoint outputs/my_experiment/checkpoints/checkpoint_step_10000 \
    --output_dir outputs/eval \
    --save_predictions
```

## Notes

- All experiments were run with mixed precision training (bfloat16) for efficiency
- Checkpoints are saved every 500 steps, with automatic cleanup keeping only the latest 3
- Best model is selected based on validation loss
- Training time: ~8-12 hours for 10K steps on A100 40GB
- Evaluation time: ~30-60 minutes depending on dataset size

## Troubleshooting

If you encounter issues reproducing results:

1. **Check CUDA version compatibility**
2. **Verify dataset paths and formats**
3. **Ensure random seed is set correctly**
4. **Check that all dependencies match versions**
5. **Verify GPU memory is sufficient**

See [Troubleshooting Guide](troubleshooting.md) for more details.
