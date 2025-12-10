# Confidence Prediction Training Pipeline

A research pipeline for training LLMs to predict step-level confidence scores using a special `<CONFIDENCE>` token and MLP head.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start training with LoRA (recommended)
python train_lora.py --experiment_name my_model --max_steps 10000

# Or full fine-tuning
python train.py --experiment_name my_model --max_steps 10000
```

## Documentation

Complete documentation is in [`docs/`](docs/index.md)

### Core Guides
- **[Setup Guide](docs/setup.md)** - Installation and configuration
- **[Quick Start](docs/quickstart.md)** - Get training in 5 minutes
- **[Training Guide](docs/training.md)** - Complete training documentation
- **[Evaluation Guide](docs/evaluation.md)** - Metrics and analysis

### Reference
- **[Architecture](docs/architecture.md)** - Model architecture and design
- **[Dataset Guide](docs/datasets.md)** - Dataset information and label mappings
- **[Configuration](docs/configuration.md)** - All configuration options
- **[API Reference](docs/api-reference.md)** - Code documentation

### Help
- **[Troubleshooting](docs/troubleshooting.md)** - Common issues and solutions
- **[Best Practices](docs/best-practices.md)** - Optimization strategies

## Project Structure

```
/workspace/Main/
├── docs/                    # Complete documentation
├── models/                  # Model implementations
│   ├── mlp_head.py         # Standalone MLP head
│   ├── model.py            # Full fine-tuning model
│   └── lora_model.py       # LoRA model
├── data/                    # Data loaders
│   ├── dataset.py          # Unified dataset
│   ├── prm800k_loader.py   # PRM800K loader
│   ├── reveal_loader.py    # REVEAL loader
│   └── eqasc_loader.py     # eQASC loader
├── utils/                   # Utilities
│   ├── metrics.py          # Evaluation metrics
│   └── logger.py           # Logging
├── outputs/                 # Training outputs (created during training)
│   └── {experiment_name}/
│       ├── checkpoints/    # Saved model checkpoints
│       └── logs/           # Training logs
├── train.py                 # Full fine-tuning training
├── train_lora.py           # LoRA training
├── evaluate.py             # Evaluation script
├── config.py               # Configuration system
└── requirements.txt        # Dependencies
```

## Overview

This pipeline enables LLMs to predict confidence scores (0-1 scale) for each reasoning step. The model is trained on three complementary datasets:

- **PRM800K** (97K examples) - Math reasoning with correctness ratings
- **REVEAL** (6K examples) - Multi-domain QA with evidence attribution
- **eQASC** (119K chains) - Science QA with relevance scores

### Key Features

- **Multi-dataset training** - Combines three datasets with weighted sampling
- **Dual training modes** - Full fine-tuning and LoRA support
- **Comprehensive metrics** - MSE, MAE, R², ECE, stratified MAE, per-dataset evaluation
- **Production-ready** - Modular architecture, checkpointing, logging
- **Well-documented** - Complete guides in [`docs/`](docs/index.md)

## Quick Examples

### Training

```bash
# LoRA training (recommended)
python train_lora.py --experiment_name my_exp --max_steps 10000

# Full fine-tuning
python train.py --experiment_name my_exp --max_steps 10000

# Custom configuration
python train_lora.py --config my_config.json
```

### Evaluation

```bash
python evaluate.py \
    --checkpoint outputs/my_exp/checkpoints/checkpoint_step_10000 \
    --output_dir outputs/eval \
    --save_predictions
```

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended: A100 40GB+)
- PyTorch 2.0+
- Transformers 4.40+
- See [requirements.txt](requirements.txt) for full list

## License

MIT License

---

**Get started**: Read the [Quick Start Guide](docs/quickstart.md) or see [complete documentation](docs/index.md).
