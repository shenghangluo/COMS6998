# Architecture Overview

## System Architecture

The confidence prediction pipeline consists of three main components:

```
Input Text → LLM Backbone → <CONFIDENCE> Token Embedding → MLP Head → Confidence Score [0, 1]
```

## Components

### 1. LLM Backbone

**Model**: Qwen/Qwen3-4B-Instruct-2507
- 4 billion parameters
- Instruction-tuned for reasoning tasks
- Hidden size: 4096 dimensions

**Modifications**:
- Special `<CONFIDENCE>` token added to vocabulary
- Token embeddings resized to accommodate new token
- Original vocab size + 1 token

### 2. MLP Head

**Architecture**:
```
Input: 4096-dim embedding (from <CONFIDENCE> token)
  ↓
Linear(4096 → 1024) + GELU + Dropout(0.1)
  ↓
Linear(1024 → 256) + GELU + Dropout(0.1)
  ↓
Linear(256 → 1) + Sigmoid
  ↓
Output: Single confidence score [0, 1]
```

**Design Decisions**:
- **Regression output**: Predicts continuous values [0, 1] rather than classification
- **Progressive dimension reduction**: 4096 → 1024 → 256 → 1
- **GELU activation**: Better gradient flow than ReLU
- **Dropout regularization**: 0.1 dropout rate to prevent overfitting
- **Modular design**: MLP head is separate module, reusable across training modes

**Implementation**: See [models/mlp_head.py](../models/mlp_head.py)

### 3. Confidence Token Integration

**Training Format**:
```
Problem: Calculate 2+2.

Step 1: Add the numbers together. <CONFIDENCE>
Step 2: The result is 4. <CONFIDENCE>
```

**How It Works**:
1. Input text with `<CONFIDENCE>` tokens is tokenized
2. Forward pass through LLM backbone produces hidden states
3. Embeddings at `<CONFIDENCE>` token positions are extracted
4. Each embedding is passed through MLP head
5. MLP outputs confidence score for that step

**Position Tracking**:
- Automatic detection of `<CONFIDENCE>` token positions in sequence
- Handles variable number of steps per example
- Padding for batch processing

## Training Modes

### Full Fine-Tuning

**What's Trained**:
- All LLM backbone parameters (4B params)
- MLP head parameters (~5M params)
- New confidence token embedding

**Memory**: ~40GB GPU VRAM

**Learning Rate**: 2e-5 (lower due to large parameter count)

**Use When**:
- Maximum performance needed
- Sufficient GPU memory available
- Adapting model to new domain

**Implementation**: See [models/model.py](../models/model.py) and [train.py](../train.py)

### LoRA (Low-Rank Adaptation)

**What's Trained**:
- LoRA adapters on attention layers (~1% of backbone params)
- MLP head parameters (~5M params)
- New confidence token embedding

**Memory**: ~20GB GPU VRAM (50% reduction)

**Learning Rate**: 1e-4 (higher due to fewer parameters)

**LoRA Configuration**:
```python
r = 16              # Rank of low-rank matrices
alpha = 32          # Scaling factor (alpha/r = 2.0)
target_modules = [  # Which layers get adapters
    "q_proj",       # Query projection
    "k_proj",       # Key projection
    "v_proj",       # Value projection
    "o_proj"        # Output projection
]
dropout = 0.1       # LoRA dropout
```

**Use When**:
- Memory constrained
- Faster iteration needed
- Similar performance to full fine-tuning acceptable

**Implementation**: See [models/lora_model.py](../models/lora_model.py) and [train_lora.py](../train_lora.py)

## Training Objective

### Loss Function

**Mean Squared Error (MSE)**:
```python
loss = MSE(predictions, labels)
     = mean((predictions - labels)²)
```

**Why MSE?**
- Regression task (predicting continuous confidence scores)
- Penalizes large errors heavily
- Differentiable for gradient descent

**Alternative**: Huber loss (configurable, more robust to outliers)

### Optimization

**Optimizer**: AdamW8bit (8-bit AdamW from bitsandbytes)
```python
lr = 2e-5 (full) or 1e-4 (LoRA)
betas = (0.9, 0.999)
weight_decay = 0.01
```

Note: 8-bit optimizer states save ~75% memory compared to float32 (8GB vs 32GB for 4B params).

**Learning Rate Schedule**: Cosine with warmup
- Warmup: 500 steps (linear increase)
- Decay: Cosine annealing to 0

**Gradient Clipping**: max_norm = 1.0

**Mixed Precision**: bfloat16 for memory efficiency

## Data Flow

### Training Step

1. **Batch Construction**:
   ```
   [Problem + Steps + <CONFIDENCE> tokens] → Tokenize → Pad to max_length
   ```

2. **Forward Pass**:
   ```
   Input IDs → LLM Backbone → Hidden States
                              ↓
   Extract at <CONFIDENCE> positions → [batch, num_steps, 4096]
                                        ↓
   MLP Head → [batch, num_steps, 1] → Predictions
   ```

3. **Loss Calculation**:
   ```
   MSE(predictions, ground_truth_labels)
   ```
   Only valid positions (non-padded) contribute to loss.

4. **Backward Pass**:
   ```
   Gradients flow through MLP → Token embeddings → (LoRA adapters or full backbone)
   ```

### Inference

1. Generate text with model
2. Insert `<CONFIDENCE>` tokens after each reasoning step
3. Run forward pass to get confidence scores
4. Optionally: use scores to filter/rank reasoning paths

## Model Files

### Structure

**Full Fine-Tuning Checkpoint**:
```
checkpoint_step_N/
├── backbone.pt          # Full model weights (~8GB)
├── mlp_head.pt          # MLP head weights (~20MB)
├── optimizer.pt         # Optimizer state
├── scheduler.pt         # LR scheduler state
└── config.json          # Training configuration
```

**LoRA Checkpoint**:
```
checkpoint_step_N/
├── lora_adapter/        # LoRA weights (~200MB)
│   ├── adapter_config.json
│   └── adapter_model.bin
├── mlp_head.pt          # MLP head weights (~20MB)
├── optimizer.pt
├── scheduler.pt
└── config.json
```

### Loading Models

```python
from models.model import ConfidenceModel

# Full fine-tuning
model = ConfidenceModel(...)
model.backbone.load_state_dict(torch.load("backbone.pt"))
model.mlp_head.load_state_dict(torch.load("mlp_head.pt"))

# LoRA (adapters loaded automatically by PEFT)
from models.lora_model import ConfidenceModelLoRA
model = ConfidenceModelLoRA(...)
model.load_mlp_head("mlp_head.pt")
```

## Design Rationale

### Why Separate MLP Head?

**Benefits**:
- **Reusable**: Same MLP architecture for full and LoRA training
- **Modular**: Easy to experiment with different head architectures
- **Debuggable**: Can test MLP independently
- **Efficient**: Small parameter count relative to backbone

### Why Regression vs Classification?

**Regression [0, 1]**:
- ✅ Nuanced confidence levels (e.g., 0.75 vs 0.80)
- ✅ Natural interpretation as probability/confidence
- ✅ Flexible thresholds in downstream use

**Classification (e.g., High/Medium/Low)**:
- ❌ Loss of granularity
- ❌ Arbitrary bin boundaries
- ❌ Less flexible

### Why Multi-Dataset Training?

**Advantages**:
- Better generalization across domains
- More training data
- Complementary signal types (correctness, attribution, relevance)

**Implementation**: Weighted sampling ensures balanced learning from all datasets.

## Performance Characteristics

### Memory Usage

| Configuration | GPU Memory | Training Speed |
|--------------|------------|----------------|
| Full (batch=4) | ~40GB | Baseline |
| Full (batch=2) | ~25GB | 0.5× speed |
| LoRA (batch=4) | ~20GB | 1.3× speed |
| LoRA (batch=2) | ~15GB | 0.65× speed |

### Throughput

- **Full fine-tuning**: ~5-8 samples/sec (A100)
- **LoRA**: ~7-10 samples/sec (A100)
- **Effective batch size**: batch_size × gradient_accumulation = 4 × 4 = 16

### Convergence

- **Full fine-tuning**: Better final metrics, slower training
- **LoRA**: 90-95% of full performance, 30-40% faster

## Code Organization

```
models/
├── mlp_head.py         # Standalone MLP (ConfidenceMLP class)
├── model.py            # Full model (ConfidenceModel class)
└── lora_model.py       # LoRA model (ConfidenceModelLoRA class)

data/
├── dataset.py          # Unified dataset (ConfidenceDataset class)
├── prm800k_loader.py   # Data loading functions
├── reveal_loader.py
└── eqasc_loader.py

utils/
├── metrics.py          # Evaluation functions
└── logger.py           # Logging utilities
```

## Further Reading

- **[Training Guide](training.md)** - How to train models
- **[Dataset Guide](datasets.md)** - Dataset details and label mappings
- **[API Reference](api-reference.md)** - Code documentation
- **[Best Practices](best-practices.md)** - Optimization strategies
