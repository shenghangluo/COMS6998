"""Unified dataset class for confidence prediction across multiple datasets."""

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import List, Dict, Optional, Tuple
import random

from .prm800k_loader import load_prm800k
from .reveal_loader import load_reveal
from .eqasc_loader import load_eqasc


class ConfidenceDataset(Dataset):
    """
    Unified dataset for training confidence prediction models.

    Combines PRM800K, REVEAL, and eQASC datasets with proper tokenization
    and formatting for step-level confidence prediction.

    Args:
        tokenizer: HuggingFace tokenizer
        dataset_configs: List of dataset configurations, each with:
            - name: 'prm800k', 'reveal', or 'eqasc'
            - path: Path to dataset directory
            - split: 'train' or 'test'/'eval'/'dev'
            - weight: Sampling weight (optional)
            - max_samples: Max samples to load (optional)
        confidence_token: Special token to insert after steps (default: "<CONFIDENCE>")
        max_length: Maximum sequence length (default: 2048)
        max_steps_per_example: Maximum steps to include per example (default: 10)
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        dataset_configs: List[Dict],
        confidence_token: str = "<CONFIDENCE>",
        max_length: int = 2048,
        max_steps_per_example: int = 10,
    ):
        self.tokenizer = tokenizer
        self.confidence_token = confidence_token
        self.max_length = max_length
        self.max_steps_per_example = max_steps_per_example

        # Ensure confidence token is in tokenizer
        if confidence_token not in tokenizer.get_vocab():
            raise ValueError(
                f"Confidence token '{confidence_token}' not in tokenizer vocabulary. "
                f"Add it using tokenizer.add_special_tokens()"
            )

        self.confidence_token_id = tokenizer.convert_tokens_to_ids(confidence_token)

        # Load all datasets
        self.examples = []
        self.dataset_weights = []

        for config in dataset_configs:
            dataset_name = config['name']
            dataset_path = config['path']
            split = config.get('split', 'train')
            weight = config.get('weight', 1.0)
            max_samples = config.get('max_samples', None)

            print(f"\nLoading {dataset_name} ({split})...")

            # Load dataset
            if dataset_name == 'prm800k':
                phase = config.get('phase', 2)
                examples = load_prm800k(dataset_path, split=split, phase=phase, max_samples=max_samples)
            elif dataset_name == 'reveal':
                examples = load_reveal(dataset_path, split=split, max_samples=max_samples)
            elif dataset_name == 'eqasc':
                examples = load_eqasc(dataset_path, split=split, max_samples=max_samples)
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")

            # Add to unified dataset
            self.examples.extend(examples)
            self.dataset_weights.extend([weight] * len(examples))

        print(f"\nTotal examples loaded: {len(self.examples)}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training example.

        Returns:
            Dict with:
                - input_ids: Tokenized input with <CONFIDENCE> tokens
                - attention_mask: Attention mask
                - confidence_positions: Positions of <CONFIDENCE> tokens
                - labels: Ground truth confidence scores
                - source: Dataset source
        """
        example = self.examples[idx]

        # Format example text with confidence tokens
        formatted_text = self._format_example(example)

        # Tokenize
        encoding = self.tokenizer(
            formatted_text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        # Find positions of confidence tokens
        confidence_positions = self._find_confidence_positions(input_ids)

        # Get ground truth labels (pad to max_steps)
        confidences = example['confidences'][:self.max_steps_per_example]

        # Validate all confidences are in [0, 1] range
        for i, conf in enumerate(confidences):
            if conf < 0 or conf > 1:
                raise ValueError(
                    f"Confidence value {conf} out of [0, 1] range in {example['source']} dataset "
                    f"at example {idx}, step {i}. All confidence scores must be in [0, 1] range."
                )

        labels = torch.full((self.max_steps_per_example,), -1.0, dtype=torch.float32)
        labels[:len(confidences)] = torch.tensor(confidences, dtype=torch.float32)

        # Ensure confidence_positions matches labels length
        if len(confidence_positions) < self.max_steps_per_example:
            # Pad with -1
            padded_positions = torch.full((self.max_steps_per_example,), -1, dtype=torch.long)
            padded_positions[:len(confidence_positions)] = confidence_positions
            confidence_positions = padded_positions
        else:
            confidence_positions = confidence_positions[:self.max_steps_per_example]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'confidence_positions': confidence_positions,
            'labels': labels,
            'source': example['source']
        }

    def _format_example(self, example: Dict) -> str:
        """
        Format an example as text with <CONFIDENCE> tokens.

        Args:
            example: Example dict from one of the loaders

        Returns:
            Formatted string ready for tokenization
        """
        source = example['source']

        if source == 'prm800k':
            # Format: Problem + step-by-step solution
            text = f"Problem: {example['problem']}\n\nSolution:\n"
            for i, step in enumerate(example['steps'][:self.max_steps_per_example]):
                text += f"{step} {self.confidence_token}\n"

        elif source == 'reveal':
            # Format: Question + reasoning steps
            text = f"Question: {example['question']}\n\nReasoning:\n"
            for i, step in enumerate(example['steps'][:self.max_steps_per_example]):
                text += f"{step} {self.confidence_token}\n"

        elif source == 'eqasc':
            # Format: Question + answer + reasoning chain
            text = f"Question: {example['question']}\n"
            text += f"Answer: {example['answer']}\n\nReasoning:\n"
            for i, step in enumerate(example['steps'][:self.max_steps_per_example]):
                text += f"{step} {self.confidence_token}\n"

        else:
            raise ValueError(f"Unknown source: {source}")

        return text

    def _find_confidence_positions(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Find positions of confidence tokens in tokenized sequence."""
        positions = (input_ids == self.confidence_token_id).nonzero(as_tuple=True)[0]
        return positions

    def get_weighted_sampler(self):
        """
        Get a weighted sampler for balanced multi-dataset training.

        Returns:
            torch.utils.data.WeightedRandomSampler
        """
        from torch.utils.data import WeightedRandomSampler

        # Normalize weights
        total_weight = sum(self.dataset_weights)
        normalized_weights = [w / total_weight for w in self.dataset_weights]

        sampler = WeightedRandomSampler(
            weights=normalized_weights,
            num_samples=len(self.examples),
            replacement=True
        )

        return sampler


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching.

    Args:
        batch: List of examples from __getitem__

    Returns:
        Batched tensors
    """
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'confidence_positions': torch.stack([item['confidence_positions'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch]),
        'source': [item['source'] for item in batch]
    }


if __name__ == "__main__":
    # Test dataset
    from transformers import AutoTokenizer

    print("Testing ConfidenceDataset...")

    # Load tokenizer and add confidence token
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507", trust_remote_code=True)
    tokenizer.add_special_tokens({'additional_special_tokens': ['<CONFIDENCE>']})

    # Create dataset configs (small samples for testing)
    dataset_configs = [
        {
            'name': 'prm800k',
            'path': '/workspace/datasets/PRM800K',
            'split': 'train',
            'phase': 2,
            'weight': 0.5,
            'max_samples': 5
        },
        {
            'name': 'reveal',
            'path': '/workspace/datasets/reveal',
            'split': 'eval',
            'weight': 0.25,
            'max_samples': 5
        },
        {
            'name': 'eqasc',
            'path': '/workspace/datasets/eQASC',
            'split': 'train',
            'weight': 0.25,
            'max_samples': 5
        }
    ]

    # Create dataset
    dataset = ConfidenceDataset(
        tokenizer=tokenizer,
        dataset_configs=dataset_configs,
        max_length=512,
        max_steps_per_example=5
    )

    print(f"\nDataset size: {len(dataset)}")

    # Test first example
    example = dataset[0]
    print(f"\nFirst example:")
    print(f"  Source: {example['source']}")
    print(f"  Input shape: {example['input_ids'].shape}")
    print(f"  Confidence positions: {example['confidence_positions']}")
    print(f"  Labels: {example['labels']}")

    # Decode to see formatted text
    decoded = tokenizer.decode(example['input_ids'], skip_special_tokens=False)
    print(f"\nFormatted text:\n{decoded[:500]}...")
