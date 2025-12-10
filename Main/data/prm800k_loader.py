"""Data loader for PRM800K dataset (Process Reward Model)."""

import json
from typing import List, Dict, Tuple
from pathlib import Path


def load_prm800k(
    data_path: str,
    split: str = "train",
    phase: int = 2,
    max_samples: int = None
) -> List[Dict]:
    """
    Load PRM800K dataset and extract step-level reasoning with confidence labels.

    Label mapping (0-1 scale):
        rating +1 (correct) -> 1.0
        rating 0 (neutral) -> 0.5
        rating -1 (incorrect) -> 0.0

    Args:
        data_path: Path to PRM800K dataset directory
        split: "train" or "test"
        phase: 1 or 2 (phase 2 is larger and more complex)
        max_samples: Maximum number of samples to load (None for all)

    Returns:
        List of dicts with format:
        {
            'problem': str,
            'steps': List[str],  # List of reasoning steps
            'confidences': List[float],  # Confidence scores (0-1)
            'ground_truth': str,
            'source': 'prm800k'
        }
    """
    data_path = Path(data_path)
    file_name = f"phase{phase}_{split}.jsonl"
    file_path = data_path / file_name

    if not file_path.exists():
        raise FileNotFoundError(f"PRM800K file not found: {file_path}")

    print(f"Loading PRM800K from {file_path}...")

    # Rating to confidence mapping (normalized to [0, 1] range)
    rating_to_confidence = {
        1: 1.0,     # Correct step
        0: 0.5,     # Neutral/uncertain
        -1: 0.0     # Incorrect step
    }
    print(f"[DEBUG] PRM800K rating_to_confidence mapping: {rating_to_confidence}")

    examples = []
    skipped_none_ratings = 0
    skipped_unexpected_ratings = 0

    with open(file_path, 'r') as f:
        for idx, line in enumerate(f):
            if max_samples and idx >= max_samples:
                break

            data = json.loads(line)

            # Extract problem and ground truth
            problem = data['question']['problem']
            ground_truth = data['question']['ground_truth_answer']

            # Extract steps and their ratings
            steps = []
            confidences = []

            for step_data in data['label']['steps']:
                # Get the chosen completion
                chosen_idx = step_data.get('chosen_completion')

                if chosen_idx is None or chosen_idx < 0:
                    # Use human completion if available
                    if step_data.get('human_completion') and step_data['human_completion'].get('text'):
                        step_text = step_data['human_completion']['text']
                        rating = step_data['human_completion'].get('rating')
                    else:
                        continue
                else:
                    # Use chosen completion
                    completions = step_data.get('completions', [])
                    if chosen_idx >= len(completions):
                        continue

                    completion = completions[chosen_idx]
                    step_text = completion['text']
                    rating = completion.get('rating')

                # Handle None ratings (unrated steps)
                if rating is None:
                    skipped_none_ratings += 1
                    continue

                # Convert rating to confidence [0, 1]
                if rating not in rating_to_confidence:
                    skipped_unexpected_ratings += 1
                    continue
                confidence = rating_to_confidence[rating]

                steps.append(step_text)
                confidences.append(confidence)

            # Only include examples with at least one step
            if steps:
                examples.append({
                    'problem': problem,
                    'steps': steps,
                    'confidences': confidences,
                    'ground_truth': ground_truth,
                    'source': 'prm800k'
                })

    print(f"Loaded {len(examples)} examples from PRM800K {split} set (phase {phase})")
    if skipped_none_ratings > 0:
        print(f"  Skipped {skipped_none_ratings} steps with None ratings")
    if skipped_unexpected_ratings > 0:
        print(f"  Skipped {skipped_unexpected_ratings} steps with unexpected ratings")

    return examples


def format_prm800k_example(example: Dict, include_confidence: bool = False) -> str:
    """
    Format a PRM800K example as text.

    Args:
        example: Example dict from load_prm800k
        include_confidence: Whether to include <CONFIDENCE> tokens

    Returns:
        Formatted string
    """
    text = f"Problem: {example['problem']}\n\nSolution:\n"

    for i, step in enumerate(example['steps']):
        text += f"Step {i+1}: {step}"
        if include_confidence:
            text += f" <CONFIDENCE>"
        text += "\n"

    return text


if __name__ == "__main__":
    # Test loading PRM800K
    dataset_path = "/workspace/datasets/PRM800K"

    # Load a small sample
    train_examples = load_prm800k(dataset_path, split="train", phase=2, max_samples=5)

    print(f"\nExample 1:")
    print(f"Problem: {train_examples[0]['problem']}")
    print(f"Number of steps: {len(train_examples[0]['steps'])}")
    print(f"Steps and confidences:")
    for step, conf in zip(train_examples[0]['steps'], train_examples[0]['confidences']):
        print(f"  - [{conf:.0f}] {step[:100]}...")

    print(f"\nFormatted example with confidence tokens:")
    print(format_prm800k_example(train_examples[0], include_confidence=True))
