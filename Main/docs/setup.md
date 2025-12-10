# Setup Guide

## Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended: A100 with 40GB+ VRAM)
- 50GB+ free disk space

## Installation

### 1. Install Dependencies

```bash
cd /workspace/Main
pip install -r requirements.txt
```

### Core Dependencies
- `torch>=2.0.0` - PyTorch framework
- `transformers>=4.40.0` - HuggingFace transformers
- `peft>=0.7.0` - Parameter-efficient fine-tuning (LoRA)
- `bitsandbytes` - 8-bit optimizer support
- `numpy`, `scikit-learn`, `tqdm` - Utilities

See [requirements.txt](../requirements.txt) for the complete list.

### 2. Verify Dataset Access

Ensure datasets are available at:
- `/workspace/datasets/PRM800K/` - Math reasoning dataset
- `/workspace/datasets/reveal/` - Attribution dataset
- `/workspace/datasets/eQASC/` - Science QA dataset

### 3. Verify Installation

Run a quick test to ensure everything is set up correctly:

```bash
python train_lora.py --experiment_name quick_test --max_steps 100
```

This runs a 100-step training to validate:
- Dependencies are installed
- Datasets are accessible
- GPU is available and configured
- Model can be loaded and trained

**Expected output**: Test should complete in 5-10 minutes and create outputs in `outputs/quick_test/`.

## Directory Structure

After installation, your project structure should look like:

```
/workspace/Main/
├── docs/            # Documentation (this directory)
├── models/          # Model implementations
├── data/            # Data loaders
├── utils/           # Utilities (metrics, logging)
├── outputs/         # Created during training
│   └── {experiment_name}/
│       ├── checkpoints/
│       └── logs/
├── train.py         # Full fine-tuning training
├── train_lora.py    # LoRA training
├── evaluate.py      # Evaluation script
├── config.py        # Configuration system
└── requirements.txt # Dependencies
```

## Configuration

### Default Configuration

The default configuration in [config.py](../config.py) uses:
- Model: Qwen/Qwen3-4B-Instruct-2507
- Learning rate: 2e-5 (full) / 1e-4 (LoRA)
- Batch size: 4 with 4x gradient accumulation
- All three datasets with weighted sampling

### Custom Configuration

Create a custom config:

```python
from config import Config

config = Config()
config.experiment_name = "my_experiment"
config.training.learning_rate = 5e-5
config.training.max_steps = 5000
config.save("my_config.json")
```

Then use it:

```bash
python train.py --config my_config.json
```

## Next Steps

- **[Quick Start](quickstart.md)** - Run your first training
- **[Training Guide](training.md)** - Comprehensive training documentation
- **[Configuration Reference](configuration.md)** - All configuration options

## Troubleshooting

### GPU Not Detected

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If `False`, check CUDA installation.

### Out of Memory

Reduce batch size or use LoRA:

```bash
# For full fine-tuning
python train.py --batch_size 2

# For LoRA (uses less memory by default)
python train_lora.py --experiment_name my_exp --max_steps 10000
```

See [Troubleshooting Guide](troubleshooting.md) for more solutions.
