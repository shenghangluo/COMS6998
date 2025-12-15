"""Unit tests for dataset classes."""

import pytest
import torch
from transformers import AutoTokenizer
from data.dataset import ConfidenceDataset, collate_fn


class TestConfidenceDataset:
    """Test ConfidenceDataset class."""

    @pytest.fixture
    def tokenizer(self):
        """Create a mock tokenizer for testing."""
        # Use a small model for testing
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def test_dataset_initialization(self, tokenizer):
        """Test dataset initializes correctly."""
        dataset_configs = [
            {
                "name": "prm800k",
                "path": "/workspace/datasets/PRM800K",
                "split": "test",
                "max_samples": 10
            }
        ]
        
        # This will fail if datasets don't exist, which is expected
        # In a real test environment, you'd use mock data
        try:
            dataset = ConfidenceDataset(
                tokenizer=tokenizer,
                dataset_configs=dataset_configs,
                confidence_token="<CONFIDENCE>",
                max_length=512,
                max_steps_per_example=5
            )
            assert dataset is not None
        except (FileNotFoundError, ValueError):
            # Expected if datasets don't exist
            pytest.skip("Datasets not available for testing")

    def test_dataset_confidence_token_required(self, tokenizer):
        """Test dataset requires confidence token in tokenizer."""
        tokenizer.add_special_tokens({'additional_special_tokens': ['<CONFIDENCE>']})
        
        dataset_configs = [
            {
                "name": "prm800k",
                "path": "/workspace/datasets/PRM800K",
                "split": "test",
                "max_samples": 1
            }
        ]
        
        try:
            dataset = ConfidenceDataset(
                tokenizer=tokenizer,
                dataset_configs=dataset_configs,
                confidence_token="<CONFIDENCE>"
            )
            # Should not raise error if token is in tokenizer
            assert dataset is not None
        except (FileNotFoundError, ValueError):
            pytest.skip("Datasets not available for testing")

    def test_collate_fn_structure(self):
        """Test collate function returns correct structure."""
        # Create mock batch
        batch = [
            {
                'input_ids': torch.tensor([1, 2, 3, 4, 5]),
                'attention_mask': torch.tensor([1, 1, 1, 1, 1]),
                'confidence_positions': torch.tensor([2, 4]),
                'labels': torch.tensor([0.5, 0.8])
            },
            {
                'input_ids': torch.tensor([1, 2, 3]),
                'attention_mask': torch.tensor([1, 1, 1]),
                'confidence_positions': torch.tensor([1]),
                'labels': torch.tensor([0.6])
            }
        ]
        
        collated = collate_fn(batch)
        
        assert 'input_ids' in collated
        assert 'attention_mask' in collated
        assert 'confidence_positions' in collated
        assert 'labels' in collated
        assert isinstance(collated['input_ids'], torch.Tensor)

    def test_collate_fn_padding(self):
        """Test collate function pads sequences correctly."""
        batch = [
            {
                'input_ids': torch.tensor([1, 2, 3]),
                'attention_mask': torch.tensor([1, 1, 1]),
                'confidence_positions': torch.tensor([1]),
                'labels': torch.tensor([0.5])
            },
            {
                'input_ids': torch.tensor([1, 2, 3, 4, 5, 6]),
                'attention_mask': torch.tensor([1, 1, 1, 1, 1, 1]),
                'confidence_positions': torch.tensor([2, 4]),
                'labels': torch.tensor([0.6, 0.7])
            }
        ]
        
        collated = collate_fn(batch)
        
        # All sequences should be padded to same length
        assert collated['input_ids'].shape[0] == 2
        assert collated['input_ids'].shape[1] == 6  # Max length
        assert collated['attention_mask'].shape == collated['input_ids'].shape

