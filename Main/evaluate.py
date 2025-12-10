"""Evaluation script for confidence prediction models."""

import os
import sys
import torch
import numpy as np
import json
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import argparse
from collections import defaultdict

# Add LLM directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models.model import ConfidenceModel
from models.lora_model import ConfidenceModelLoRA
from data.dataset import ConfidenceDataset, collate_fn
from utils.metrics import compute_all_metrics, compute_per_dataset_metrics
from config import Config


def load_model(checkpoint_path: str, config: Config, tokenizer, use_lora: bool = False):
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint directory
        config: Configuration object
        tokenizer: Tokenizer
        use_lora: Whether model uses LoRA

    Returns:
        Loaded model
    """
    print(f"Loading model from {checkpoint_path}...")

    checkpoint_path = Path(checkpoint_path)

    # Get torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }
    torch_dtype = dtype_map.get(config.model.torch_dtype, torch.bfloat16)

    if use_lora:
        # Load LoRA model
        model = ConfidenceModelLoRA(
            model_name=config.model.model_name,
            lora_config={
                "r": config.lora.lora_r,
                "lora_alpha": config.lora.lora_alpha,
                "lora_dropout": config.lora.lora_dropout,
                "target_modules": config.lora.lora_target_modules
            },
            mlp_config={
                "input_dim": None,
                "hidden_dims": config.model.mlp_hidden_dims,
                "dropout": config.model.mlp_dropout,
                "activation": config.model.mlp_activation
            },
            device_map=config.model.device_map,
            torch_dtype=torch_dtype
        )

        # Load LoRA adapter and MLP head
        lora_adapter_path = checkpoint_path / "lora_adapter"
        mlp_head_path = checkpoint_path / "mlp_head.pt"

        if lora_adapter_path.exists() and mlp_head_path.exists():
            # Note: PEFT loads adapter differently
            model.load_mlp_head(str(mlp_head_path))
        else:
            print("Warning: LoRA checkpoint files not found, using freshly initialized weights")

    else:
        # Load full fine-tuned model
        model = ConfidenceModel(
            model_name=config.model.model_name,
            mlp_config={
                "input_dim": None,
                "hidden_dims": config.model.mlp_hidden_dims,
                "dropout": config.model.mlp_dropout,
                "activation": config.model.mlp_activation
            },
            device_map=config.model.device_map,
            torch_dtype=torch_dtype
        )

        # Load checkpoint weights
        backbone_path = checkpoint_path / "backbone.pt"
        mlp_head_path = checkpoint_path / "mlp_head.pt"

        if backbone_path.exists() and mlp_head_path.exists():
            print("PATCH: Resizing embeddings to 151670 (Checkpoint Size)...")
            model.backbone.resize_token_embeddings(151670)

            print("PATCH: Loading weights with prefix fix...")
            # Fix Backbone Keys
            bb_state = torch.load(backbone_path, map_location='cpu')
            fixed_bb = {k.replace("model.", ""): v for k, v in bb_state.items()}
            model.backbone.load_state_dict(fixed_bb, strict=False)

            # Fix Head Keys (CORRECTED: Keep 'mlp.' prefix)
            print("Loading head weights directly...")
            head_state = torch.load(mlp_head_path, map_location='cpu')
            # Only remove 'mlp_head.' if it exists (rare), but keep 'mlp.'
            fixed_head = {k.replace("mlp_head.", ""): v for k, v in head_state.items()}
            model.mlp_head.load_state_dict(fixed_head)
        else:
            print("Warning: Checkpoint files not found, using freshly initialized weights")

    # Resize token embeddings and set confidence token ID
    model.backbone.resize_token_embeddings(len(tokenizer))
    confidence_token_id = tokenizer.convert_tokens_to_ids(config.model.confidence_token)
    model.set_confidence_token_id(confidence_token_id)

    model.eval()

    print("Model loaded successfully!")

    return model


@torch.no_grad()
def evaluate_model(model, dataloader, save_predictions: bool = False):
    """
    Evaluate model and collect predictions.

    Args:
        model: Model to evaluate
        dataloader: Dataloader
        save_predictions: Whether to save individual predictions

    Returns:
        Tuple of (metrics, predictions_dict)
    """
    device = next(model.parameters()).device

    all_predictions = []
    all_labels = []
    all_sources = []
    predictions_list = [] if save_predictions else None

    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        print(f"Processing Batch {batch_idx+1}/10...")
        if batch_idx >= 10: break # PATCH: Fast Exit
        if batch_idx >= 50: break  # PATCH: Fast Exit
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        confidence_positions = batch['confidence_positions'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            confidence_positions=confidence_positions,
            labels=labels,
            return_dict=True
        )

        predictions = outputs['predictions']

        # Collect for metrics
        for i in range(predictions.size(0)):
            valid_mask = labels[i] >= 0
            if valid_mask.any():
                pred_valid = predictions[i][valid_mask].float().cpu().numpy()
                label_valid = labels[i][valid_mask].float().cpu().numpy()

                all_predictions.append(pred_valid)
                all_labels.append(label_valid)
                all_sources.append(batch['source'][i])

                # Save individual predictions if requested
                if save_predictions:
                    predictions_list.append({
                        'source': batch['source'][i],
                        'predictions': pred_valid.tolist(),
                        'labels': label_valid.tolist()
                    })

        if outputs['loss'] is not None:
            total_loss += outputs['loss'].item()
            num_batches += 1

    # Concatenate all predictions and labels
    all_predictions_concat = np.concatenate(all_predictions)
    all_labels_concat = np.concatenate(all_labels)

    # Compute overall metrics
    metrics = compute_all_metrics(all_predictions_concat, all_labels_concat)
    metrics['loss'] = total_loss / num_batches if num_batches > 0 else 0.0

    # Compute per-dataset metrics
    dataset_names = list(set(all_sources))
    per_dataset_metrics = {}

    for dataset_name in dataset_names:
        dataset_preds = []
        dataset_labels = []

        for pred, label, source in zip(all_predictions, all_labels, all_sources):
            if source == dataset_name:
                dataset_preds.append(pred)
                dataset_labels.append(label)

        if dataset_preds:
            dataset_preds_concat = np.concatenate(dataset_preds)
            dataset_labels_concat = np.concatenate(dataset_labels)
            per_dataset_metrics[dataset_name] = compute_all_metrics(
                dataset_preds_concat, dataset_labels_concat
            )

    return {
        'overall': metrics,
        'per_dataset': per_dataset_metrics
    }, predictions_list


def print_metrics(metrics: dict, title: str = "Evaluation Results"):
    """Pretty print evaluation metrics."""
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")

    # Overall metrics
    print("\nOverall Metrics:")
    overall = metrics.get('overall', {})
    for key, value in overall.items():
        if key == 'n_samples':
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value:.4f}")

    # Per-dataset metrics
    per_dataset = metrics.get('per_dataset', {})
    if per_dataset:
        print("\nPer-Dataset Metrics:")
        for dataset_name, dataset_metrics in per_dataset.items():
            print(f"\n  {dataset_name.upper()}:")
            for key, value in dataset_metrics.items():
                if key == 'n_samples':
                    print(f"    {key}: {value}")
                else:
                    print(f"    {key}: {value:.4f}")

    print(f"\n{'='*60}\n")


def main(args):
    """Main evaluation function."""
    # Load config
    if args.config:
        config = Config.from_json(args.config)
    else:
        # Try to load from checkpoint
        checkpoint_config = Path(args.checkpoint) / "config.json"
        if checkpoint_config.exists():
            config = Config.from_json(str(checkpoint_config))
        else:
            print("Warning: No config found, using default config")
            from config import get_default_config
            config = get_default_config()

    # Prepare tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.model_name,
        trust_remote_code=True
    )
    special_tokens = {'additional_special_tokens': [config.model.confidence_token]}
    tokenizer.add_special_tokens(special_tokens)

    # Prepare dataset
    print("\nPreparing evaluation dataset...")
    eval_dataset_configs = config.get_dataset_configs("test")

    eval_dataset = ConfidenceDataset(
        tokenizer=tokenizer,
        dataset_configs=eval_dataset_configs,
        confidence_token=config.model.confidence_token,
        max_length=config.data.max_length,
        max_steps_per_example=config.data.max_steps_per_example
    )

    print(f"Evaluation dataset size: {len(eval_dataset)}")

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    # Load model
    model = load_model(
        args.checkpoint,
        config,
        tokenizer,
        use_lora=config.lora.use_lora
    )

    # Evaluate
    print("\nRunning evaluation...")
    metrics, predictions = evaluate_model(
        model,
        eval_dataloader,
        save_predictions=args.save_predictions
    )

    # Print results
    print_metrics(metrics, title="Evaluation Results")

    # Save results
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics
        metrics_file = output_dir / "eval_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {metrics_file}")

        # Save predictions if requested
        if args.save_predictions and predictions:
            predictions_file = output_dir / "predictions.json"
            with open(predictions_file, 'w') as f:
                json.dump(predictions, f, indent=2)
            print(f"Predictions saved to {predictions_file}")

    print("\nEvaluation completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate confidence prediction model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default=None, help="Path to config file (optional)")
    parser.add_argument("--output_dir", type=str, default="/workspace/logs/eval", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--save_predictions", action="store_true", help="Save individual predictions")

    args = parser.parse_args()

    main(args)
