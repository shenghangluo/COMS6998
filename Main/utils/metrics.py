"""Metrics for evaluating confidence prediction models."""

import numpy as np
from typing import Dict, List, Tuple
import torch


def compute_mse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute Mean Squared Error.

    Args:
        predictions: Predicted confidence scores
        targets: Ground truth confidence scores

    Returns:
        MSE value
    """
    return float(np.mean((predictions - targets) ** 2))


def compute_mae(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute Mean Absolute Error.

    Args:
        predictions: Predicted confidence scores
        targets: Ground truth confidence scores

    Returns:
        MAE value
    """
    return float(np.mean(np.abs(predictions - targets)))


def compute_r2(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute R² (coefficient of determination).

    Args:
        predictions: Predicted confidence scores
        targets: Ground truth confidence scores

    Returns:
        R² value (1.0 is perfect, 0.0 is baseline, negative is worse than baseline)
    """
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)

    if ss_tot == 0:
        return 0.0

    return float(1 - (ss_res / ss_tot))


def compute_ece(
    predictions: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 10,
    score_range: Tuple[float, float] = (0, 1)
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    Measures how well the predicted confidence scores match actual accuracy.
    For regression, we bin by predicted confidence and compare to actual values.

    Args:
        predictions: Predicted confidence scores
        targets: Ground truth confidence scores
        n_bins: Number of bins to use
        score_range: Range of scores (min, max)

    Returns:
        ECE value (lower is better, 0 is perfect calibration)
    """
    min_score, max_score = score_range
    bin_width = (max_score - min_score) / n_bins

    ece = 0.0
    total_samples = len(predictions)

    for i in range(n_bins):
        bin_lower = min_score + i * bin_width
        bin_upper = bin_lower + bin_width

        # Find samples in this bin
        in_bin = (predictions >= bin_lower) & (predictions < bin_upper)

        if i == n_bins - 1:  # Include upper bound in last bin
            in_bin = (predictions >= bin_lower) & (predictions <= bin_upper)

        n_in_bin = np.sum(in_bin)

        if n_in_bin > 0:
            # Average predicted confidence in bin
            avg_confidence = np.mean(predictions[in_bin])

            # Average actual confidence in bin
            avg_actual = np.mean(targets[in_bin])

            # Weighted contribution to ECE
            ece += (n_in_bin / total_samples) * abs(avg_confidence - avg_actual)

    return float(ece)


def compute_rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute Root Mean Squared Error.

    Args:
        predictions: Predicted confidence scores
        targets: Ground truth confidence scores

    Returns:
        RMSE value
    """
    return float(np.sqrt(compute_mse(predictions, targets)))


def compute_pearson_correlation(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute Pearson correlation coefficient.

    Args:
        predictions: Predicted confidence scores
        targets: Ground truth confidence scores

    Returns:
        Correlation coefficient (-1 to 1)
    """
    if len(predictions) < 2:
        return 0.0

    return float(np.corrcoef(predictions, targets)[0, 1])


def compute_stratified_mae(
    predictions: np.ndarray,
    targets: np.ndarray
) -> Dict[str, float]:
    """
    Compute MAE stratified by confidence ranges.

    Critical for multi-modal distributions where overall MAE can hide
    poor performance in specific confidence ranges.

    Args:
        predictions: Predicted confidence scores
        targets: Ground truth confidence scores

    Returns:
        Dict with MAE for each confidence range:
        - 'mae_low': MAE for targets in [0, 0.25]
        - 'mae_medium': MAE for targets in (0.25, 0.75)
        - 'mae_high': MAE for targets in [0.75, 1.0]
    """
    stratified_metrics = {}

    # Define confidence ranges
    ranges = {
        'mae_low': (0.0, 0.25),
        'mae_medium': (0.25, 0.75),
        'mae_high': (0.75, 1.0)
    }

    for metric_name, (low, high) in ranges.items():
        # Create mask for this range (inclusive on both ends)
        if metric_name == 'mae_low':
            mask = (targets >= low) & (targets <= high)
        elif metric_name == 'mae_medium':
            mask = (targets > low) & (targets < high)
        else:  # mae_high
            mask = (targets >= low) & (targets <= high)

        if mask.any():
            mae = float(np.mean(np.abs(predictions[mask] - targets[mask])))
            stratified_metrics[metric_name] = mae
        else:
            # No samples in this range
            stratified_metrics[metric_name] = float('nan')

    return stratified_metrics


def compute_all_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 10,
    score_range: Tuple[float, float] = (0, 1)
) -> Dict[str, float]:
    """
    Compute all regression and calibration metrics.

    Args:
        predictions: Predicted confidence scores
        targets: Ground truth confidence scores
        n_bins: Number of bins for ECE
        score_range: Range of scores

    Returns:
        Dict with metric values:
        - mse: Mean Squared Error
        - mae: Mean Absolute Error (overall)
        - r2: Coefficient of Determination
        - ece: Expected Calibration Error
        - mae_low: MAE for low confidence [0, 0.25]
        - mae_medium: MAE for medium confidence (0.25, 0.75)
        - mae_high: MAE for high confidence [0.75, 1.0]
        - n_samples: Number of samples
    """
    # Handle empty inputs
    if len(predictions) == 0 or len(targets) == 0:
        return {
            'mse': float('inf'),
            'mae': float('inf'),
            'r2': -float('inf'),
            'ece': float('inf'),
            'mae_low': float('nan'),
            'mae_medium': float('nan'),
            'mae_high': float('nan'),
            'n_samples': 0
        }

    # Convert to numpy if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    # Flatten if needed
    predictions = predictions.flatten()
    targets = targets.flatten()

    # Compute base metrics
    metrics = {
        'mse': compute_mse(predictions, targets),
        'mae': compute_mae(predictions, targets),
        'r2': compute_r2(predictions, targets),
        'ece': compute_ece(predictions, targets, n_bins, score_range),
        'n_samples': len(predictions)
    }

    # Add stratified MAE metrics
    stratified = compute_stratified_mae(predictions, targets)
    metrics.update(stratified)

    return metrics


def compute_per_dataset_metrics(
    predictions: List[np.ndarray],
    targets: List[np.ndarray],
    sources: List[str],
    dataset_names: List[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics separately for each dataset source.

    Args:
        predictions: List of prediction arrays
        targets: List of target arrays
        sources: List of dataset source identifiers
        dataset_names: Optional list of unique dataset names to compute for

    Returns:
        Dict mapping dataset name to metrics dict
    """
    if dataset_names is None:
        dataset_names = list(set(sources))

    per_dataset_metrics = {}

    for dataset_name in dataset_names:
        # Filter predictions/targets for this dataset
        dataset_preds = []
        dataset_targets = []

        for pred, target, source in zip(predictions, targets, sources):
            if source == dataset_name:
                dataset_preds.append(pred)
                dataset_targets.append(target)

        if dataset_preds:
            # Concatenate all predictions/targets for this dataset
            all_preds = np.concatenate(dataset_preds)
            all_targets = np.concatenate(dataset_targets)

            # Compute metrics
            per_dataset_metrics[dataset_name] = compute_all_metrics(all_preds, all_targets)
        else:
            per_dataset_metrics[dataset_name] = {
                'mse': float('inf'),
                'mae': float('inf'),
                'rmse': float('inf'),
                'r2': -float('inf'),
                'ece': float('inf'),
                'correlation': 0.0,
                'n_samples': 0
            }

    return per_dataset_metrics


if __name__ == "__main__":
    # Test metrics
    print("Testing metrics...")

    # Create dummy data
    np.random.seed(42)
    predictions = np.random.uniform(0, 1, 1000)
    targets = predictions + np.random.normal(0, 0.1, 1000)  # Add noise
    targets = np.clip(targets, 0, 1)  # Clip to valid range

    # Compute all metrics
    metrics = compute_all_metrics(predictions, targets)

    print("\nMetrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    # Test perfect predictions
    perfect_preds = np.array([0, 0.25, 0.5, 0.75, 1.0])
    perfect_targets = np.array([0, 0.25, 0.5, 0.75, 1.0])

    perfect_metrics = compute_all_metrics(perfect_preds, perfect_targets)
    print("\nPerfect prediction metrics:")
    for name, value in perfect_metrics.items():
        print(f"  {name}: {value:.4f}")
