"""Data loader for REVEAL dataset (attribution and reasoning)."""

import json
from typing import List, Dict
from pathlib import Path


def load_reveal(
    data_path: str,
    split: str = "eval",
    max_samples: int = None
) -> List[Dict]:
    """
    Load REVEAL dataset and extract reasoning steps with attribution-based confidence.

    Label mapping (0-1 scale):
        "Fully" supported -> 1.0
        "Partially" supported -> 0.5
        "Unsupported" -> 0.25
        "Contradictory" -> 0.0

    Args:
        data_path: Path to REVEAL dataset directory
        split: "eval" or "open"
        max_samples: Maximum number of samples to load (None for all)

    Returns:
        List of dicts with format:
        {
            'question': str,
            'steps': List[str],  # List of reasoning steps
            'confidences': List[float],  # Confidence scores (0-1)
            'evidence': List[str],  # Evidence for each step
            'source': 'reveal'
        }
    """
    data_path = Path(data_path)
    file_name = f"{split}.json"
    file_path = data_path / file_name

    if not file_path.exists():
        raise FileNotFoundError(f"REVEAL file not found: {file_path}")

    print(f"Loading REVEAL from {file_path}...")

    # Attribution label to confidence mapping (normalized to [0, 1] range)
    attribution_to_confidence = {
        "Fully": 1.0,
        "Partially": 0.5,
        "Unsupported": 0.25,
        "Contradictory": 0.0
    }

    # Group by question_id to reconstruct full reasoning chains
    questions_dict = {}

    with open(file_path, 'r') as f:
        for idx, line in enumerate(f):
            if max_samples and idx >= max_samples:
                break

            data = json.loads(line)

            # Skip if no attribution label
            if not data.get('attribution_label'):
                continue

            question_id = data.get('question_id', f"unknown_{idx}")
            question = data.get('question', '')
            step = data.get('step', '')
            evidence = data.get('evidence', '')
            attribution_label = data.get('attribution_label', 'Unsupported')

            # Convert attribution to confidence
            confidence = attribution_to_confidence.get(attribution_label, 0.25)

            # Group by question_id
            if question_id not in questions_dict:
                questions_dict[question_id] = {
                    'question': question,
                    'steps': [],
                    'confidences': [],
                    'evidence': [],
                    'source': 'reveal'
                }

            questions_dict[question_id]['steps'].append(step)
            questions_dict[question_id]['confidences'].append(confidence)
            questions_dict[question_id]['evidence'].append(evidence)

    # Convert dict to list and filter out questions with no steps
    examples = [
        example for example in questions_dict.values()
        if example['steps']
    ]

    print(f"Loaded {len(examples)} examples from REVEAL {split} set")

    # Print statistics
    total_steps = sum(len(ex['steps']) for ex in examples)
    avg_steps = total_steps / len(examples) if examples else 0
    print(f"  Total steps: {total_steps}, Average steps per question: {avg_steps:.2f}")

    return examples


def format_reveal_example(example: Dict, include_confidence: bool = False) -> str:
    """
    Format a REVEAL example as text.

    Args:
        example: Example dict from load_reveal
        include_confidence: Whether to include <CONFIDENCE> tokens

    Returns:
        Formatted string
    """
    text = f"Question: {example['question']}\n\nReasoning:\n"

    for i, (step, evidence) in enumerate(zip(example['steps'], example.get('evidence', []))):
        text += f"Step {i+1}: {step}"
        if include_confidence:
            text += f" <CONFIDENCE>"
        text += f"\nEvidence: {evidence[:200]}...\n\n"

    return text


def compute_reveal_statistics(examples: List[Dict]) -> Dict:
    """
    Compute statistics about REVEAL dataset labels.

    Args:
        examples: List of examples from load_reveal

    Returns:
        Dict with label distribution statistics
    """
    from collections import Counter

    # Map confidences back to labels for statistics
    confidence_to_label = {
        1.0: "Fully",
        0.5: "Partially",
        0.25: "Unsupported",
        0.0: "Contradictory"
    }

    all_confidences = []
    for example in examples:
        all_confidences.extend(example['confidences'])

    label_counts = Counter(confidence_to_label.get(c, "Unknown") for c in all_confidences)
    total = sum(label_counts.values())

    stats = {
        'total_steps': total,
        'label_distribution': {
            label: {
                'count': count,
                'percentage': (count / total * 100) if total > 0 else 0
            }
            for label, count in label_counts.items()
        }
    }

    return stats


if __name__ == "__main__":
    # Test loading REVEAL
    dataset_path = "/workspace/datasets/reveal"

    # Load eval set
    eval_examples = load_reveal(dataset_path, split="eval", max_samples=100)

    print(f"\nExample 1:")
    print(f"Question: {eval_examples[0]['question']}")
    print(f"Number of steps: {len(eval_examples[0]['steps'])}")
    print(f"Steps and confidences:")
    for step, conf, evidence in zip(
        eval_examples[0]['steps'],
        eval_examples[0]['confidences'],
        eval_examples[0]['evidence']
    ):
        print(f"  - [{conf:.0f}] {step[:80]}...")
        print(f"    Evidence: {evidence[:100]}...")

    # Compute statistics
    print(f"\nDataset Statistics:")
    stats = compute_reveal_statistics(eval_examples)
    print(f"Total steps: {stats['total_steps']}")
    print(f"Label distribution:")
    for label, info in stats['label_distribution'].items():
        print(f"  {label}: {info['count']} ({info['percentage']:.2f}%)")
