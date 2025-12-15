"""Unit tests for metrics computation."""

import pytest
import numpy as np
from utils.metrics import (
    compute_mse,
    compute_mae,
    compute_r2,
    compute_ece,
    compute_stratified_mae,
    compute_all_metrics,
    compute_per_dataset_metrics
)


class TestBasicMetrics:
    """Test basic regression metrics."""

    def test_mse_perfect_predictions(self):
        """Test MSE with perfect predictions."""
        predictions = np.array([0.0, 0.5, 1.0])
        targets = np.array([0.0, 0.5, 1.0])
        mse = compute_mse(predictions, targets)
        assert mse == 0.0

    def test_mse_with_errors(self):
        """Test MSE with prediction errors."""
        predictions = np.array([0.1, 0.5, 0.9])
        targets = np.array([0.0, 0.5, 1.0])
        mse = compute_mse(predictions, targets)
        assert mse > 0.0
        assert abs(mse - 0.0067) < 0.001  # (0.1^2 + 0.1^2) / 3

    def test_mae_perfect_predictions(self):
        """Test MAE with perfect predictions."""
        predictions = np.array([0.0, 0.5, 1.0])
        targets = np.array([0.0, 0.5, 1.0])
        mae = compute_mae(predictions, targets)
        assert mae == 0.0

    def test_mae_with_errors(self):
        """Test MAE with prediction errors."""
        predictions = np.array([0.1, 0.5, 0.9])
        targets = np.array([0.0, 0.5, 1.0])
        mae = compute_mae(predictions, targets)
        assert mae > 0.0
        assert abs(mae - 0.0667) < 0.001  # (0.1 + 0.1) / 3

    def test_r2_perfect_predictions(self):
        """Test R² with perfect predictions."""
        predictions = np.array([0.0, 0.5, 1.0])
        targets = np.array([0.0, 0.5, 1.0])
        r2 = compute_r2(predictions, targets)
        assert abs(r2 - 1.0) < 1e-6

    def test_r2_baseline(self):
        """Test R² equals zero when predictions equal mean."""
        targets = np.array([0.0, 0.5, 1.0])
        predictions = np.array([0.5, 0.5, 0.5])  # Mean of targets
        r2 = compute_r2(predictions, targets)
        assert abs(r2) < 1e-6

    def test_r2_negative(self):
        """Test R² can be negative when worse than baseline."""
        predictions = np.array([1.0, 0.0, 1.0])
        targets = np.array([0.0, 1.0, 0.0])
        r2 = compute_r2(predictions, targets)
        assert r2 < 0.0


class TestCalibrationMetrics:
    """Test calibration metrics."""

    def test_ece_perfect_calibration(self):
        """Test ECE with perfectly calibrated predictions."""
        # Perfect calibration: predictions match actual confidence
        predictions = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        targets = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        ece = compute_ece(predictions, targets, n_bins=5, score_range=(0, 1))
        assert ece < 0.01  # Should be very close to zero

    def test_ece_miscalibrated(self):
        """Test ECE with miscalibrated predictions."""
        # Predictions are too high
        predictions = np.array([0.9, 0.9, 0.9, 0.9, 0.9])
        targets = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        ece = compute_ece(predictions, targets, n_bins=5, score_range=(0, 1))
        assert ece > 0.0

    def test_ece_range(self):
        """Test ECE is in valid range [0, 1]."""
        predictions = np.random.uniform(0, 1, 100)
        targets = np.random.uniform(0, 1, 100)
        ece = compute_ece(predictions, targets, n_bins=10, score_range=(0, 1))
        assert 0.0 <= ece <= 1.0


class TestStratifiedMetrics:
    """Test stratified MAE metrics."""

    def test_stratified_mae_structure(self):
        """Test stratified MAE returns correct structure."""
        predictions = np.random.uniform(0, 1, 100)
        targets = np.random.uniform(0, 1, 100)
        result = compute_stratified_mae(predictions, targets)
        
        assert 'mae_low' in result
        assert 'mae_medium' in result
        assert 'mae_high' in result
        assert 'n_samples' in result
        assert all(isinstance(v, (int, float)) for v in result.values())

    def test_stratified_mae_ranges(self):
        """Test stratified MAE uses correct ranges."""
        # Create data in each range
        predictions = np.array([0.1, 0.5, 0.9, 0.2, 0.6, 0.8])
        targets = np.array([0.0, 0.5, 1.0, 0.25, 0.75, 0.85])
        
        result = compute_stratified_mae(predictions, targets)
        
        # Check that all ranges have samples
        assert result['n_samples'] > 0
        assert result['mae_low'] >= 0
        assert result['mae_medium'] >= 0
        assert result['mae_high'] >= 0


class TestAllMetrics:
    """Test compute_all_metrics function."""

    def test_compute_all_metrics_structure(self):
        """Test compute_all_metrics returns all expected metrics."""
        predictions = np.random.uniform(0, 1, 100)
        targets = np.random.uniform(0, 1, 100)
        metrics = compute_all_metrics(predictions, targets)
        
        expected_keys = ['mse', 'mae', 'r2', 'ece', 'mae_low', 'mae_medium', 
                        'mae_high', 'n_samples']
        for key in expected_keys:
            assert key in metrics, f"Missing metric: {key}"

    def test_compute_all_metrics_values(self):
        """Test compute_all_metrics returns valid values."""
        predictions = np.array([0.1, 0.5, 0.9])
        targets = np.array([0.0, 0.5, 1.0])
        metrics = compute_all_metrics(predictions, targets)
        
        assert metrics['mse'] >= 0
        assert metrics['mae'] >= 0
        assert metrics['n_samples'] == 3
        assert -1.0 <= metrics['r2'] <= 1.0
        assert 0.0 <= metrics['ece'] <= 1.0


class TestPerDatasetMetrics:
    """Test per-dataset metrics computation."""

    def test_per_dataset_metrics_structure(self):
        """Test per-dataset metrics returns correct structure."""
        predictions = [np.array([0.5, 0.7]), np.array([0.3, 0.9])]
        targets = [np.array([0.5, 0.8]), np.array([0.2, 1.0])]
        sources = ['prm800k', 'reveal']
        
        result = compute_per_dataset_metrics(predictions, targets, sources)
        
        assert 'prm800k' in result
        assert 'reveal' in result
        assert 'overall' in result

    def test_per_dataset_metrics_empty(self):
        """Test per-dataset metrics with empty data."""
        predictions = []
        targets = []
        sources = []
        
        result = compute_per_dataset_metrics(predictions, targets, sources)
        assert isinstance(result, dict)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_arrays(self):
        """Test metrics with empty arrays."""
        predictions = np.array([])
        targets = np.array([])
        
        # Should handle gracefully or raise appropriate error
        with pytest.raises((ValueError, ZeroDivisionError)):
            compute_mse(predictions, targets)

    def test_different_lengths(self):
        """Test metrics with mismatched array lengths."""
        predictions = np.array([0.5, 0.7])
        targets = np.array([0.5, 0.7, 0.9])
        
        # Should raise ValueError
        with pytest.raises(ValueError):
            compute_mse(predictions, targets)

    def test_nan_handling(self):
        """Test metrics handle NaN values."""
        predictions = np.array([0.5, np.nan, 0.9])
        targets = np.array([0.5, 0.7, 0.9])
        
        # Should handle NaN or raise appropriate error
        with pytest.raises((ValueError, RuntimeWarning)):
            compute_mse(predictions, targets)

    def test_inf_handling(self):
        """Test metrics handle Inf values."""
        predictions = np.array([0.5, np.inf, 0.9])
        targets = np.array([0.5, 0.7, 0.9])
        
        # Should handle Inf or raise appropriate error
        with pytest.raises((ValueError, RuntimeWarning)):
            compute_mse(predictions, targets)

