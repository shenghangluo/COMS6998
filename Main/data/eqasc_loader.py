"""Data loader for eQASC dataset (Explanatory QA with Structured Chains)."""

import json
from typing import List, Dict
from pathlib import Path


def load_eqasc(
    data_path: str,
    split: str = "train",
    max_samples: int = None,
    min_score: float = 10.0,
    max_score: float = 88.0
) -> List[Dict]:
    """
    Load eQASC dataset and extract reasoning chains with scores as confidence.

    The scores are on a scale from ~10-88 and are scaled using min-max normalization
    to map to the full [0, 1] range for better alignment with other datasets.

    Min-max scaling: (score - min_score) / (max_score - min_score)

    Args:
        data_path: Path to eQASC dataset directory
        split: "train", "dev", or "test"
        max_samples: Maximum number of samples to load (None for all)
        min_score: Minimum score for normalization (default: 10.0)
        max_score: Maximum score for normalization (default: 88.0)

    Returns:
        List of dicts with format:
        {
            'question': str,
            'answer': str,
            'steps': List[str],  # Reasoning chain steps
            'confidences': List[float],  # Min-max scaled scores in [0, 1] range
            'source': 'eqasc'
        }
    """
    data_path = Path(data_path)

    # Map split name to file name
    split_to_file = {
        "train": "eqasc_train_grc.json",
        "dev": "eqasc_dev_grc.json",
        "test": "eqasc_test_grc.json"
    }

    file_name = split_to_file.get(split)
    if not file_name:
        raise ValueError(f"Invalid split: {split}. Choose from {list(split_to_file.keys())}")

    file_path = data_path / file_name

    if not file_path.exists():
        raise FileNotFoundError(f"eQASC file not found: {file_path}")

    print(f"Loading eQASC from {file_path}...")

    with open(file_path, 'r') as f:
        data = json.load(f)

    examples = []

    # Process each question
    for idx, item in enumerate(data):
        if max_samples and idx >= max_samples:
            break

        question_stem = item.get('question', {}).get('stem', '')
        answer_key = item.get('answerKey', '')

        # Get choices
        choices = item.get('question', {}).get('choices', [])

        # Process each choice
        for choice in choices:
            choice_label = choice.get('label', '')
            choice_text = choice.get('text', '')

            # Get reasoning chains for this choice
            chains = choice.get('chains', [])

            # Create an example for each reasoning chain
            for chain in chains:
                # Each chain is a list of facts (dictionaries with 'text' and 'score')
                if not isinstance(chain, list):
                    continue

                steps = []
                confidences = []

                # Extract text and score from each fact in the chain
                for fact in chain:
                    if isinstance(fact, dict) and 'text' in fact:
                        fact_text = fact.get('text', '').strip()
                        fact_score = fact.get('score', 0.0)

                        if fact_text:
                            # Min-max scaling to [0, 1] range
                            scaled_score = (fact_score - min_score) / (max_score - min_score)

                            # Clamp to [0, 1] in case of outliers
                            if scaled_score < 0 or scaled_score > 1:
                                print(f"WARNING: eQASC score {fact_score} (scaled: {scaled_score:.4f}) out of [0, 1], clamping")
                                scaled_score = max(0.0, min(1.0, scaled_score))

                            steps.append(fact_text)
                            confidences.append(scaled_score)

                if steps:
                    examples.append({
                        'question': question_stem,
                        'answer': f"{choice_label}. {choice_text}",
                        'is_correct': (choice_label == answer_key),
                        'steps': steps,
                        'confidences': confidences,
                        'source': 'eqasc'
                    })

    print(f"Loaded {len(examples)} reasoning chains from eQASC {split} set")

    # Print statistics
    total_steps = sum(len(ex['steps']) for ex in examples)
    avg_steps = total_steps / len(examples) if examples else 0
    print(f"  Total steps: {total_steps}, Average steps per chain: {avg_steps:.2f}")

    # Print score statistics
    all_scores = [conf for ex in examples for conf in ex['confidences']]
    if all_scores:
        print(f"  Score range: {min(all_scores):.2f} - {max(all_scores):.2f}")
        print(f"  Mean score: {sum(all_scores)/len(all_scores):.2f}")

    return examples


def format_eqasc_example(example: Dict, include_confidence: bool = False) -> str:
    """
    Format an eQASC example as text.

    Args:
        example: Example dict from load_eqasc
        include_confidence: Whether to include <CONFIDENCE> tokens

    Returns:
        Formatted string
    """
    text = f"Question: {example['question']}\n"
    text += f"Answer choice: {example['answer']}\n\nReasoning:\n"

    for i, step in enumerate(example['steps']):
        text += f"Step {i+1}: {step}"
        if include_confidence:
            text += f" <CONFIDENCE>"
        text += "\n"

    return text


def filter_eqasc_by_correctness(examples: List[Dict], correct_only: bool = True) -> List[Dict]:
    """
    Filter eQASC examples by whether they lead to correct answers.

    Args:
        examples: List of examples from load_eqasc
        correct_only: If True, keep only chains for correct answers

    Returns:
        Filtered list of examples
    """
    filtered = [ex for ex in examples if ex.get('is_correct', False) == correct_only]
    print(f"Filtered to {len(filtered)} examples (correct_only={correct_only})")
    return filtered


def normalize_eqasc_scores(examples: List[Dict], target_range: tuple = (0, 1)) -> List[Dict]:
    """
    Normalize eQASC scores to a target range (default: 0-1).

    Args:
        examples: List of examples from load_eqasc
        target_range: Tuple of (min, max) for target range

    Returns:
        Examples with normalized confidence scores
    """
    # Find min and max scores in the dataset
    all_scores = [conf for ex in examples for conf in ex['confidences']]
    if not all_scores:
        return examples

    min_score = min(all_scores)
    max_score = max(all_scores)

    target_min, target_max = target_range

    # Normalize each example
    normalized_examples = []
    for example in examples:
        normalized_confidences = [
            target_min + (conf - min_score) / (max_score - min_score) * (target_max - target_min)
            for conf in example['confidences']
        ]

        normalized_example = example.copy()
        normalized_example['confidences'] = normalized_confidences
        normalized_examples.append(normalized_example)

    print(f"Normalized scores from [{min_score:.2f}, {max_score:.2f}] to [{target_min}, {target_max}]")

    return normalized_examples


if __name__ == "__main__":
    # Test loading eQASC
    dataset_path = "/workspace/datasets/eQASC"

    # Load a small sample
    train_examples = load_eqasc(dataset_path, split="train", max_samples=10)

    print(f"\nExample 1:")
    print(f"Question: {train_examples[0]['question']}")
    print(f"Answer: {train_examples[0]['answer']}")
    print(f"Is correct: {train_examples[0]['is_correct']}")
    print(f"Number of steps: {len(train_examples[0]['steps'])}")
    print(f"Steps and confidences:")
    for step, conf in zip(train_examples[0]['steps'], train_examples[0]['confidences']):
        print(f"  - [{conf:.2f}] {step}")

    print(f"\nFormatted example:")
    print(format_eqasc_example(train_examples[0], include_confidence=True))

    # Test normalization (note: scores are already in [0, 1] range by default)
    print(f"\n\nTesting normalization (already scaled to [0, 1]):")
    print(f"Score range: {train_examples[0]['confidences'][0]:.2f}")
    print(f"All scores are already scaled to [0, 1] range via division by 100")
