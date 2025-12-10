# Dataset Guide

## Overview

The pipeline trains on three complementary datasets, each providing different types of confidence signals:

| Dataset | Domain | Size | Label Type | Confidence Signal |
|---------|--------|------|------------|-------------------|
| **PRM800K** | Math | 97K examples | Correctness ratings | Step-level accuracy |
| **REVEAL** | Multi-domain | 6K examples | Attribution labels | Evidence support |
| **eQASC** | Science QA | 119K chains | Relevance scores | Chain quality |

## Label Mapping to [0, 1] Scale

All datasets are normalized to a unified **[0, 1] confidence scale** for regression training.

### PRM800K (Math Reasoning)

**Original Labels**: Step correctness ratings
- `+1` - Correct step
- `0` - Neutral/uncertain step
- `-1` - Incorrect step

**Mapped to**:
- `+1` → **1.0** (fully confident)
- `0` → **0.5** (uncertain)
- `-1` → **0.0** (not confident)

**Example**:
```
Problem: How many seconds are in 7.8 minutes?

Step 1: "7.8 minutes = 7 minutes + 0.8 minutes"     [Rating: +1 → 1.0]
Step 2: "7 minutes = 7 × 60 = 420 seconds"          [Rating: +1 → 1.0]
Step 3: "0.8 minutes = 0.8 × 60 = 48 seconds"       [Rating: +1 → 1.0]
Step 4: "Total: 420 + 48 = 468 seconds"             [Rating: +1 → 1.0]
```

**Statistics**:
- Training: 97,782 examples (phase 2)
- Test: 2,762 examples
- Label distribution: 68% positive, 24% negative, 9% neutral
- Average steps per example: ~8

**Location**: `/workspace/datasets/PRM800K/phase2_train.jsonl`

### REVEAL (Evidence Attribution)

**Original Labels**: Evidence support classification
- `Fully` - Fully supported by evidence
- `Partially` - Partially supported
- `Unsupported` - Not supported by evidence
- `Contradictory` - Contradicted by evidence

**Mapped to**:
- `Fully` → **1.0** (high confidence)
- `Partially` → **0.5** (medium confidence)
- `Unsupported` → **0.25** (low confidence)
- `Contradictory` → **0.0** (no confidence)

**Example**:
```
Question: What causes rain?

Step 1: "Rain is formed when water vapor condenses"
Evidence: "Water vapor condenses to form clouds and precipitation..."
[Attribution: Fully → 1.0]

Step 2: "This happens because of temperature differences"
Evidence: "Temperature changes affect atmospheric pressure..."
[Attribution: Partially → 0.5]
```

**Statistics**:
- Eval set: 4,956 examples (~73% labeled)
- Open set: 1,146 examples (~71% labeled)
- Label distribution: 64% Unsupported, 24% Fully, 6% Contradictory, 4% Partially
- Average steps per question: ~1-2

**Note**: ~27-30% of examples lack attribution labels (filtered out during training)

**Location**: `/workspace/datasets/reveal/eval.jsonl`, `open.jsonl`

### eQASC (Science QA)

**Original Labels**: Continuous relevance scores (10 - 88)

**Mapped to**: **Min-max normalization** to use full [0, 1] range
- Formula: `(score - 10) / (88 - 10)`
- Maps [10, 88] → [0.0, 1.0]

**Example**:
```
Question: What type of organism is commonly used in preparation of foods?

Chain 1 (Score: 32.7):
  Step 1: "Fungi are living organisms"
  Step 2: "Yeast is a type of fungi used in baking"
[Score: 32.7 → (32.7-10)/78 = 0.291]

Chain 2 (Score: 21.1):
  Step 1: "Bacteria are microorganisms"
  Step 2: "Some bacteria are used in fermentation"
[Score: 21.1 → (21.1-10)/78 = 0.142]
```

**Statistics**:
- Training: 8,134 questions → 118,765 reasoning chains
- Test: ~1,400 questions
- Score range after scaling: 0.00 - 0.96 (mean: 0.153, median: 0.136)
- Distribution: 90.4% in [0, 0.25], highly concentrated in low confidence
- Average steps per chain: ~2

**Location**: `/workspace/datasets/eQASC/eqasc_train_grc.json`

## Dataset Selection

### Use All Datasets (Default)

```python
from config import Config

config = Config()
# All datasets enabled by default
config.data.use_prm800k = True   # Math reasoning
config.data.use_reveal = True     # Attribution
config.data.use_eqasc = True      # Science QA
```

### Use Specific Datasets Only

**Math-focused** (PRM800K only):
```python
config.data.use_prm800k = True
config.data.use_reveal = False
config.data.use_eqasc = False
```

**Multi-domain** (REVEAL + eQASC):
```python
config.data.use_prm800k = False
config.data.use_reveal = True
config.data.use_eqasc = True
```

## Dataset Weighting

Control sampling frequency with weights:

```python
config.data.prm800k_weight = 0.5   # 50% of batches
config.data.reveal_weight = 0.25   # 25% of batches
config.data.eqasc_weight = 0.25    # 25% of batches
```

**Recommendations**:
- **Balanced**: 0.33 / 0.33 / 0.34 (equal sampling)
- **Math-focused**: 0.7 / 0.15 / 0.15 (emphasize PRM800K)
- **Multi-domain**: 0.2 / 0.4 / 0.4 (emphasize REVEAL + eQASC)

## Data Format

### Unified Format

All loaders convert to a common format:

```python
{
    'problem': str,           # Question or problem statement
    'steps': List[str],       # List of reasoning steps
    'confidences': List[float],  # Confidence scores [0, 1]
    'ground_truth': str,      # Expected answer (if available)
    'source': str            # 'prm800k', 'reveal', or 'eqasc'
}
```

### Tokenized Format

After tokenization with `<CONFIDENCE>` tokens:

```
Problem: Calculate 2+2.

Step 1: Add the numbers. <CONFIDENCE>
Step 2: Result is 4. <CONFIDENCE>
```

## Data Loading

### Manual Loading

```python
from data.prm800k_loader import load_prm800k
from data.reveal_loader import load_reveal
from data.eqasc_loader import load_eqasc

# Load PRM800K
prm800k_data = load_prm800k(
    data_path="/workspace/datasets/PRM800K",
    split="train",
    phase=2,
    max_samples=1000  # Limit for testing
)

# Load REVEAL
reveal_data = load_reveal(
    data_path="/workspace/datasets/reveal",
    split="eval",
    max_samples=100
)

# Load eQASC
eqasc_data = load_eqasc(
    data_path="/workspace/datasets/eQASC",
    split="train",
    max_samples=500
)
```

### Automatic (via Config)

```python
from config import Config
from transformers import AutoTokenizer
from data.dataset import ConfidenceDataset

config = Config()
tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
tokenizer.add_special_tokens({'additional_special_tokens': ['<CONFIDENCE>']})

# Unified dataset automatically loads and combines all enabled datasets
dataset = ConfidenceDataset(
    tokenizer=tokenizer,
    dataset_configs=config.get_dataset_configs("train"),
    confidence_token="<CONFIDENCE>",
    max_length=2048,
    max_steps_per_example=10
)
```

## Dataset Statistics Comparison

| Metric | PRM800K | REVEAL | eQASC |
|--------|---------|--------|-------|
| **Total examples** | 97,782 | ~6,000 | 118,765 chains |
| **Avg steps/example** | ~8 | ~1-2 | ~2 |
| **Total steps** | ~782K | ~8K | ~238K |
| **Label type** | Discrete (3 values) | Discrete (4 values) | Continuous |
| **Label variance** | High | High | Low |
| **Domain** | Math | Multi-domain | Science |
| **Quality** | Human-labeled | Human-labeled | Automatic scores |

## Sampling Strategy

The unified dataset uses **weighted random sampling**:

1. Calculate total weight: `W = w_prm800k + w_reveal + w_eqasc`
2. For each batch, sample dataset with probability: `p_i = w_i / W`
3. Sample example from selected dataset
4. Repeat until batch is full

This ensures balanced training even with different dataset sizes.

## Data Preprocessing

### Tokenization

```python
# Format with confidence tokens
text = "Problem: 2+2\n\nStep 1: Add numbers <CONFIDENCE>\nStep 2: Result is 4 <CONFIDENCE>"

# Tokenize
tokens = tokenizer(text, max_length=2048, truncation=True)

# Find <CONFIDENCE> positions
confidence_positions = (tokens['input_ids'] == confidence_token_id).nonzero()
```

### Batching

- Variable-length examples padded to `max_length`
- Variable number of steps padded to `max_steps_per_example`
- Invalid positions marked with `-1` in labels (ignored in loss)

## Customization

### Limit Dataset Size (for testing)

```python
config.data.max_train_samples = 1000  # Use only 1000 examples per dataset
config.data.max_test_samples = 200
```

### Adjust Sequence Length

```python
config.data.max_length = 1024  # Shorter sequences (saves memory)
config.data.max_steps_per_example = 5  # Fewer steps per example
```

### Change Dataset Splits

```python
# Use different splits
config.data.prm800k_phase = 1  # Phase 1 (easier problems)
config.data.reveal_train_split = "open"  # Use 'open' instead of 'eval'
```

## Data Quality Notes

### PRM800K
- ✅ High-quality human annotations
- ✅ Clear correctness labels
- ⚠️ Math-specific domain (may not generalize to other domains)

### REVEAL
- ✅ Multi-domain coverage
- ✅ Evidence-based labels (semantic confidence)
- ⚠️ ~30% missing labels
- ⚠️ Smaller dataset size

### eQASC
- ✅ Large number of examples
- ✅ Perfect class balance possible (with median threshold)
- ⚠️ Automatic scores (not human labels)
- ⚠️ Low score variance (72% in narrow range)

## Implementation Details

**Data Loaders**: See [data/](../data/) directory
- `prm800k_loader.py` - PRM800K parsing
- `reveal_loader.py` - REVEAL parsing
- `eqasc_loader.py` - eQASC parsing
- `dataset.py` - Unified PyTorch dataset

**Configuration**: See [Configuration Reference](configuration.md)

**Training**: See [Training Guide](training.md)
