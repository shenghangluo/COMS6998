# Experimental Results Summary

This is a summary document. For detailed results, see [docs/results.md](docs/results.md).

## Quick Results Overview

### Overall Performance (N = 163)

| Metric | Value  | Interpretation                                           |
| ------ | ------ | -------------------------------------------------------- |
| ECE    | 0.390  | **Critical calibration failure** - 39% average mismatch  |
| R²     | -4.374 | **Worse than baseline** - performs below mean predictor  |
| MAE    | 0.413  | **High error** - predictions off by 41 percentage points |

### Error Distribution by Label Range

| True Label   | MAE   | Predicted Tendency | Issue                           |
| ------------ | ----- | ------------------ | ------------------------------- |
| 1.0 (High)   | 0.464 | Clustered ~0.54    | **Severe underconfidence bias** |
| 0.5 (Medium) | 0.134 | Clustered ~0.50    | **Only reliable range**         |
| 0.0 (Low)    | 0.441 | Clustered ~0.44    | **Overconfidence bias**         |

### Key Findings

1. **Critical Calibration Failure**: ECE of 0.390 indicates the model cannot reliably self-assess reasoning quality
2. **Systematic Underconfidence Bias**: Model predicts ~0.54 for correct steps (true label 1.0)
3. **Failure at Extremes**: Highest errors at confidence extremes; only reliable in middle range (0.5)
4. **Multi-Domain Consistency**: Same bias pattern observed across PRM800K, REVEAL, and eQASC
5. **Unusable for Auditing**: Confidence scores cannot be used to identify model correctness

## Detailed Results

See [docs/results.md](docs/results.md) for:

- Complete metrics tables
- Per-dataset breakdowns
- Stratified performance analysis
- Error analysis
- Ablation studies
- Performance benchmarks

## Reproducing Results

See [docs/reproducibility.md](docs/reproducibility.md) for step-by-step instructions.

## Experimental Setup

See [docs/experimental-setup.md](docs/experimental-setup.md) for:

- Hardware specifications
- Software environment
- Dataset versions
- Hyperparameter settings
