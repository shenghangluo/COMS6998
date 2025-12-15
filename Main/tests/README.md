# Test Suite

This directory contains unit tests and integration tests for the confidence prediction training pipeline.

## Structure

```
tests/
├── __init__.py
├── test_metrics.py          # Tests for metrics computation
├── test_logger.py           # Tests for logging utility
├── test_config.py           # Tests for configuration system
├── test_mlp_head.py         # Tests for MLP head model
├── test_dataset.py          # Tests for dataset classes
└── integration/             # Integration tests
    ├── __init__.py
    ├── test_training.py     # End-to-end training tests
    └── test_evaluation.py   # End-to-end evaluation tests
```

## Running Tests

### Run All Tests

```bash
# From Main directory
pytest

# With verbose output
pytest -v

# With coverage (if pytest-cov installed)
pytest --cov=. --cov-report=html
```

### Run Specific Test Files

```bash
# Run only unit tests
pytest tests/test_metrics.py
pytest tests/test_logger.py
pytest tests/test_config.py

# Run only integration tests
pytest tests/integration/
```

### Run Specific Test Classes or Functions

```bash
# Run specific test class
pytest tests/test_metrics.py::TestBasicMetrics

# Run specific test function
pytest tests/test_metrics.py::TestBasicMetrics::test_mse_perfect_predictions
```

### Run Tests by Marker

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run slow tests
pytest -m slow

# Skip tests requiring GPU
pytest -m "not requires_gpu"

# Skip tests requiring datasets
pytest -m "not requires_data"
```

## Test Categories

### Unit Tests

- **test_metrics.py**: Tests for all metric computation functions
- **test_logger.py**: Tests for logging functionality
- **test_config.py**: Tests for configuration management
- **test_mlp_head.py**: Tests for MLP head model
- **test_dataset.py**: Tests for dataset loading and processing

### Integration Tests

- **test_training.py**: End-to-end training pipeline tests
- **test_evaluation.py**: End-to-end evaluation pipeline tests

## Test Markers

Tests are marked with categories:

- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.slow`: Tests that take a long time
- `@pytest.mark.requires_gpu`: Tests that need GPU
- `@pytest.mark.requires_data`: Tests that need dataset files

## Prerequisites

Tests require:

- All dependencies from `requirements.txt`
- pytest (already in requirements.txt)
- Optional: pytest-cov for coverage reports

## Notes

- Some tests may be skipped if datasets or models are not available
- Integration tests may require GPU access
- Tests use temporary directories and clean up after themselves
- Some tests use mock data to avoid requiring actual datasets

## Continuous Integration

Tests can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions
- name: Run tests
  run: |
    pip install -r requirements.txt
    pytest tests/ -v
```

## Writing New Tests

When adding new tests:

1. Follow naming convention: `test_*.py` for files, `test_*` for functions
2. Use pytest fixtures for setup/teardown
3. Mark tests appropriately (`@pytest.mark.unit`, etc.)
4. Use `pytest.skip()` for tests that can't run in current environment
5. Clean up temporary files/directories

Example:

```python
import pytest

@pytest.mark.unit
def test_my_function():
    """Test description."""
    result = my_function(input)
    assert result == expected
```
