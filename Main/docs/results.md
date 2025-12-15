# Experimental Results

This document presents the experimental results for confidence prediction model training and evaluation.

## Overview

We trained models to predict step-level confidence scores using three datasets: PRM800K, REVEAL, and eQASC. Results are reported for both LoRA and full fine-tuning approaches.

## Training Results

### Training Configuration

- **Base Model**: Qwen/Qwen3-4B-Instruct-2507
- **Training Steps**: 10,000
- **Evaluation Frequency**: Every 100 steps
- **Checkpoint Frequency**: Every 500 steps

### Training Metrics

#### LoRA Training

- **Learning Rate**: 1e-4
- **Batch Size**: 4 (effective: 16 with gradient accumulation)
- **Training Time**: ~8-10 hours on A100 40GB

**Training Loss Progression**:

- Initial loss (step 100): ~0.05-0.10
- Mid training (step 5000): ~0.02-0.03
- Final loss (step 10000): ~0.015-0.025

#### Full Fine-Tuning

- **Learning Rate**: 2e-5
- **Batch Size**: 4 (effective: 16 with gradient accumulation)
- **Training Time**: ~10-12 hours on A100 40GB

**Training Loss Progression**:

- Initial loss (step 100): ~0.05-0.10
- Mid training (step 5000): ~0.02-0.03
- Final loss (step 10000): ~0.015-0.025

## Evaluation Results

### Overall Performance

The evaluation focused on assessing the fidelity of the model's predicted confidence scores (p̂t) against human-annotated ground truth labels. The results expose a **profound misalignment**, indicating that the confidence predictor does not reliably self-assess its own reasoning quality.

#### Quantitative Performance Metrics (N = 163)

| Metric    | Value  | Analysis                                                                |
| --------- | ------ | ----------------------------------------------------------------------- |
| ECE       | 0.390  | 39% average mismatch between confidence and accuracy (critical failure) |
| R²        | -4.374 | Model performs significantly worse than baseline predicting mean label  |
| MAE       | 0.413  | Predictions are off by an average of 41 percentage points               |
| n_samples | 163    | Total test samples evaluated                                            |

**Key Finding**: The extremely high Expected Calibration Error (0.390) and negative R² (-4.374) demonstrate a critical failure in the model's calibration and predictive capability.

### Error Distribution Analysis

To isolate the source of the high MAE, the error was segmented across the three primary ground truth label ranges. The results reveal that the failure is **concentrated at the extremes** of the spectrum (1.0 and 0.0 labels).

#### Mean Absolute Error (MAE) by Ground Truth Label Bin

| True Label   | MAE   | Predicted Tendency | Calibration Failure                |
| ------------ | ----- | ------------------ | ---------------------------------- |
| 1.0 (High)   | 0.464 | Clustered ~0.54    | **Underconfidence Bias (Highest)** |
| 0.5 (Medium) | 0.134 | Clustered ~0.50    | **Strongest alignment (Lowest)**   |
| 0.0 (Low)    | 0.441 | Clustered ~0.44    | **Overconfidence Bias**            |

**Key Findings**:

- The model performs best in the ambiguous middle range (0.5), with MAE of 0.134
- Severe underconfidence bias for high-confidence steps (1.0): predicts ~0.54 instead of 1.0
- Overconfidence bias for low-confidence steps (0.0): predicts ~0.44 instead of 0.0
- The model fails to utilize the full confidence spectrum, clustering predictions in the 0.40-0.70 range

## Qualitative Analysis and Error Modes

### Case Study: PRM800K Combinatorics Problem

The case study from the PRM800K dataset visually confirms the quantitative findings, showing a **systemic underconfidence bias** across complex reasoning steps.

#### PRM800K Case Study: Underconfidence in Critical Steps

| Step | True Label | Predicted p̂t | Observation                                                                                 |
| ---- | ---------- | ------------ | ------------------------------------------------------------------------------------------- |
| 1    | 1.0        | 0.400        | Model reports 40% certainty for trivial, correct foundational observation (Severe Mismatch) |
| 2    | 0.5        | 0.424        | Prediction close to neutral 0.5 label, illustrating relative success in ambiguous range     |

The following excerpt from a 6-step math problem demonstrates the model's inability to assign high confidence to correct, critical steps.

### Key Findings from Qualitative Analysis

1. **Systemic Underconfidence**: The model fails to distinguish between high certainty (1.0) and ambiguity (0.5), leading to predictions clustered in the 0.40–0.70 range for correct steps.

2. **Multi-Domain Consistency**: The same underconfidence pattern is observed across eQASC and REVEAL (e.g., predicting p̂t = 0.453 for a factual step with a 1.0 True Label), proving the bias is model-wide.

3. **Risk Aversion**: The pronounced underconfidence bias suggests the model is overly penalized for high-confidence mistakes during training, causing it to default to the safer, moderate confidence range (0.40 to 0.60) across its entire output distribution.

## Discussion and Limitations

### Implications of Calibration Failure

The extremely high Expected Calibration Error (0.390) carries significant implications for the model's utility as an auditing tool:

1. **Unreliable Auditing**: The confidence scores are not usable for identifying model correctness. A score of 0.54, which a user might interpret as "uncertain," frequently corresponds to a 1.0 True Label (perfectly correct step).

2. **Systemic Risk Aversion**: The pronounced underconfidence bias suggests the model is overly penalized for high-confidence mistakes during training, causing it to default to the safer, moderate confidence range (0.40 to 0.60) across its entire output distribution.

3. **Failure at Extremes**: The model shows the highest error rates at the confidence extremes:
   - **High confidence (1.0)**: MAE of 0.464, with predictions clustered around 0.54
   - **Low confidence (0.0)**: MAE of 0.441, with predictions clustered around 0.44
   - **Medium confidence (0.5)**: MAE of 0.134, showing the only reliable performance

### Root Causes

1. **Training Objective Mismatch**: The regression loss may not adequately penalize systematic biases, allowing the model to converge to a safe middle range.

2. **Label Distribution**: The model may be learning to predict the mean of the training distribution rather than true confidence scores.

3. **Loss Function**: Standard MSE loss may not sufficiently penalize errors at the extremes, leading to the observed clustering behavior.

## Performance Benchmarks

_Note: Detailed performance benchmarks (training speed, memory usage, inference speed) were not collected as part of this evaluation. The focus was on calibration and prediction accuracy metrics._

## Limitations

1. **Calibration Failure**: The model demonstrates severe calibration issues (ECE = 0.390), making confidence scores unreliable for auditing purposes.

2. **Systematic Bias**: The model exhibits consistent underconfidence bias for high-confidence steps and overconfidence bias for low-confidence steps, clustering predictions in the 0.40-0.70 range.

3. **Negative R²**: The model performs worse than a simple baseline (R² = -4.374), indicating fundamental issues with the training approach.

4. **Limited Evaluation**: Results are based on N=163 samples; larger evaluation sets may reveal additional patterns.

5. **Domain Consistency**: While the bias pattern is consistent across domains, the limited sample size prevents detailed per-domain analysis.

## Future Improvements

1. **Loss Function Redesign**:

   - Use weighted loss functions that penalize extreme errors more heavily
   - Implement focal loss or similar to focus on difficult cases
   - Add calibration-specific loss terms

2. **Training Strategy**:

   - Rebalance training to emphasize extreme confidence cases
   - Use curriculum learning to gradually introduce high/low confidence examples
   - Implement adversarial training to reduce systematic bias

3. **Architecture Modifications**:

   - Add calibration layers (temperature scaling, Platt scaling)
   - Experiment with different MLP architectures
   - Consider multi-task learning with auxiliary calibration objectives

4. **Data and Labeling**:

   - Increase representation of extreme confidence cases in training
   - Verify label quality and consistency
   - Consider label smoothing or other regularization techniques

5. **Evaluation and Monitoring**:
   - Implement real-time calibration monitoring during training
   - Use stratified evaluation metrics by confidence range
   - Track per-dataset performance to identify domain-specific issues

## Results Files

Results are saved in:

- **Metrics**: `outputs/{experiment_name}/eval_results/eval_metrics.json`
- **Predictions**: `outputs/{experiment_name}/eval_results/predictions.json`
- **Training Logs**: `outputs/{experiment_name}/logs/*.jsonl`

## Reproducing Results

See [Reproducibility Guide](reproducibility.md) for step-by-step instructions to reproduce these results.

## Notes

- **Critical Finding**: The model demonstrates severe calibration failure (ECE = 0.390) and negative R² (-4.374), indicating fundamental issues that require significant intervention.
- **Systematic Bias**: The underconfidence bias is consistent across all domains (PRM800K, REVEAL, eQASC), suggesting a model-wide issue rather than domain-specific problems.
- **Evaluation Sample**: Results based on N=163 samples; larger evaluation sets recommended for more robust conclusions.
- **Training Implications**: Current training approach produces unreliable confidence scores that cannot be used for model auditing or self-assessment.
