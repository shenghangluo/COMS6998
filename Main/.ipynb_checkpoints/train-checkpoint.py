"""Main training script for confidence prediction (full fine-tuning)."""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_scheduler
import argparse
import bitsandbytes as bnb

# Add LLM directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models.model import ConfidenceModel
from data.dataset import ConfidenceDataset, collate_fn
from utils.metrics import compute_all_metrics
from utils.logger import Logger
from config import Config, get_default_config


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_tokenizer(config: Config):
    """
    Prepare tokenizer with confidence token.

    Args:
        config: Configuration object

    Returns:
        Tokenizer with confidence token added
    """
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.model_name,
        trust_remote_code=True
    )

    # Add confidence token
    special_tokens = {'additional_special_tokens': [config.model.confidence_token]}
    num_added = tokenizer.add_special_tokens(special_tokens)
    print(f"Added {num_added} special tokens: {config.model.confidence_token}")

    return tokenizer


def prepare_dataloaders(config: Config, tokenizer):
    """
    Prepare train and validation dataloaders.

    Args:
        config: Configuration object
        tokenizer: Tokenizer with confidence token

    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    print("\nPreparing datasets...")

    # Get dataset configs
    train_dataset_configs = config.get_dataset_configs("train")
    val_dataset_configs = config.get_dataset_configs("test")

    # Create datasets
    train_dataset = ConfidenceDataset(
        tokenizer=tokenizer,
        dataset_configs=train_dataset_configs,
        confidence_token=config.model.confidence_token,
        max_length=config.data.max_length,
        max_steps_per_example=config.data.max_steps_per_example
    )

    val_dataset = ConfidenceDataset(
        tokenizer=tokenizer,
        dataset_configs=val_dataset_configs,
        confidence_token=config.model.confidence_token,
        max_length=config.data.max_length,
        max_steps_per_example=config.data.max_steps_per_example
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.training.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.training.dataloader_num_workers,
        pin_memory=config.training.dataloader_pin_memory
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.training.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.training.dataloader_num_workers,
        pin_memory=config.training.dataloader_pin_memory
    )

    return train_dataloader, val_dataloader


def prepare_model(config: Config, tokenizer):
    """
    Prepare model with MLP head.

    Args:
        config: Configuration object
        tokenizer: Tokenizer

    Returns:
        Model
    """
    print("\nInitializing model...")

    # Get torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }
    torch_dtype = dtype_map.get(config.model.torch_dtype, torch.bfloat16)

    # Create model
    model = ConfidenceModel(
        model_name=config.model.model_name,
        mlp_config={
            "input_dim": None,  # Will be inferred from model
            "hidden_dims": config.model.mlp_hidden_dims,
            "dropout": config.model.mlp_dropout,
            "activation": config.model.mlp_activation
        },
        device_map=config.model.device_map,
        torch_dtype=torch_dtype
    )

    # Resize token embeddings for new confidence token
    model.backbone.resize_token_embeddings(len(tokenizer))

    # Enable gradient checkpointing to reduce memory usage
    if hasattr(model.backbone, 'gradient_checkpointing_enable'):
        model.backbone.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

    # Set confidence token ID
    confidence_token_id = tokenizer.convert_tokens_to_ids(config.model.confidence_token)
    model.set_confidence_token_id(confidence_token_id)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    return model


def prepare_optimizer_and_scheduler(config: Config, model, num_training_steps: int):
    """
    Prepare optimizer and learning rate scheduler.

    Args:
        config: Configuration object
        model: Model
        num_training_steps: Total number of training steps

    Returns:
        Tuple of (optimizer, scheduler)
    """
    # Use 8-bit AdamW optimizer to save memory
    # This stores optimizer states in 8-bit format, saving ~75% memory
    # 4B params × 2 states × 1 byte = 8GB (vs 32GB for float32)
    optimizer = bnb.optim.AdamW8bit(
        model.parameters(),
        lr=config.training.learning_rate,
        betas=(config.training.adam_beta1, config.training.adam_beta2),
        eps=config.training.adam_epsilon,
        weight_decay=config.training.weight_decay
    )

    # Create scheduler
    num_warmup_steps = config.training.warmup_steps
    if config.training.warmup_ratio > 0:
        num_warmup_steps = int(num_training_steps * config.training.warmup_ratio)

    scheduler = get_scheduler(
        name=config.training.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    print(f"\nOptimizer: AdamW8bit (lr={config.training.learning_rate})")
    print(f"Scheduler: {config.training.lr_scheduler_type} (warmup={num_warmup_steps} steps)")
    print("Optimizer states use 8-bit precision (saves ~24GB vs float32)")

    return optimizer, scheduler


def train_step(model, batch, optimizer, scheduler, config):
    """
    Perform a single training step.

    Args:
        model: Model
        batch: Batch of data
        optimizer: Optimizer
        scheduler: LR scheduler
        config: Configuration

    Returns:
        Dict of metrics
    """
    model.train()

    # Move batch to device and ensure correct dtype
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    confidence_positions = batch['confidence_positions'].to(device)
    labels = batch['labels'].to(device=device, dtype=dtype)

    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        confidence_positions=confidence_positions,
        labels=labels,
        return_dict=True
    )

    loss = outputs['loss']

    # Debug high loss batches
    if loss.item() > 10.0:
        predictions = outputs['predictions']
        valid_mask = labels >= 0
        valid_preds = predictions[valid_mask]
        valid_labels = labels[valid_mask]
        print(f"\n!!! HIGH LOSS DETECTED: {loss.item():.4f} !!!")
        print(f"Predictions range: [{valid_preds.min().item():.4f}, {valid_preds.max().item():.4f}]")
        print(f"Labels range: [{valid_labels.min().item():.4f}, {valid_labels.max().item():.4f}]")
        print(f"Sample predictions: {valid_preds[:10].cpu().tolist()}")
        print(f"Sample labels: {valid_labels[:10].cpu().tolist()}")
        print(f"ALL unique labels in batch: {sorted(set(valid_labels.cpu().tolist()))}")
        print(f"Batch sources: {batch['source']}")

    # Backward pass
    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

    # Optimizer step
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    return {
        'loss': loss.item(),
        'learning_rate': scheduler.get_last_lr()[0]
    }


@torch.no_grad()
def evaluate(model, dataloader, config):
    """
    Evaluate model on validation set.

    Args:
        model: Model
        dataloader: Validation dataloader
        config: Configuration

    Returns:
        Dict of evaluation metrics
    """
    model.eval()
    device = next(model.parameters()).device

    all_predictions = []
    all_labels = []
    all_sources = []
    total_loss = 0.0
    num_batches = 0
    total_batches = len(dataloader)

    # Limit evaluation batches if configured
    max_batches = config.eval.max_eval_batches
    if max_batches is not None:
        eval_batches = min(max_batches, total_batches)
        print(f"  Evaluating on {eval_batches} batches (out of {total_batches} total)")
    else:
        eval_batches = total_batches
        print(f"  Evaluating on all {total_batches} batches")

    for batch_idx, batch in enumerate(dataloader, 1):
        # Stop if we've reached max_eval_batches
        if max_batches is not None and batch_idx > max_batches:
            break

        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        confidence_positions = batch['confidence_positions'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            confidence_positions=confidence_positions,
            labels=labels,
            return_dict=True
        )

        # Collect predictions and labels
        predictions = outputs['predictions']  # (batch_size, max_steps)

        # Flatten and filter out padded positions (label == -1)
        for i in range(predictions.size(0)):
            valid_mask = labels[i] >= 0
            if valid_mask.any():
                all_predictions.append(predictions[i][valid_mask].float().cpu().numpy())
                all_labels.append(labels[i][valid_mask].float().cpu().numpy())
                all_sources.append(batch['source'][i])

        if outputs['loss'] is not None:
            total_loss += outputs['loss'].item()
            num_batches += 1

    # Concatenate all predictions and labels
    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)

    # Compute metrics
    metrics = compute_all_metrics(all_predictions, all_labels)
    metrics['loss'] = total_loss / num_batches if num_batches > 0 else 0.0

    return metrics


def _cleanup_old_checkpoints(checkpoint_dir: Path, keep_last: int = 3):
    """
    Remove old checkpoints, keeping only the latest N checkpoints.

    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last: Number of latest checkpoints to keep (default: 3)
    """
    import shutil

    # Find all checkpoint directories
    checkpoint_pattern = "checkpoint_step_*"
    checkpoints = sorted(
        checkpoint_dir.glob(checkpoint_pattern),
        key=lambda p: int(p.name.split('_')[-1]),  # Sort by step number
        reverse=True  # Most recent first
    )

    # Remove old checkpoints beyond keep_last
    for old_checkpoint in checkpoints[keep_last:]:
        print(f"Removing old checkpoint: {old_checkpoint.name}")
        shutil.rmtree(old_checkpoint)


def save_checkpoint(model, optimizer, scheduler, step, config, logger, metrics=None):
    """
    Save model checkpoint.

    Args:
        model: Model
        optimizer: Optimizer
        scheduler: Scheduler
        step: Current step
        config: Configuration
        logger: Logger
        metrics: Optional metrics dict
    """
    # Checkpoints go in: /workspace/outputs/{experiment_name}/checkpoints/checkpoint_step_{step}/
    experiment_dir = Path(config.output_dir) / config.experiment_name
    checkpoint_base_dir = experiment_dir / "checkpoints"
    checkpoint_base_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_base_dir / f"checkpoint_step_{step}"
    checkpoint_path.mkdir(exist_ok=True)

    # Save model
    torch.save(model.backbone.state_dict(), checkpoint_path / "backbone.pt")
    torch.save(model.mlp_head.state_dict(), checkpoint_path / "mlp_head.pt")

    # Save optimizer and scheduler
    torch.save(optimizer.state_dict(), checkpoint_path / "optimizer.pt")
    torch.save(scheduler.state_dict(), checkpoint_path / "scheduler.pt")

    # Save config
    config.save(str(checkpoint_path / "config.json"))

    # Log checkpoint
    logger.save_checkpoint_info(step, str(checkpoint_path), metrics or {})

    print(f"Checkpoint saved to {checkpoint_path}")

    # Clean up old checkpoints, keeping only the latest 3
    _cleanup_old_checkpoints(checkpoint_base_dir, keep_last=3)


def main(config: Config):
    """
    Main training function.

    Args:
        config: Configuration object
    """
    # Set seed
    set_seed(config.training.seed)

    # Setup experiment directory structure
    experiment_dir = Path(config.output_dir) / config.experiment_name
    log_dir = experiment_dir / "logs"
    checkpoint_base_dir = experiment_dir / "checkpoints"

    # Create directories
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_base_dir.mkdir(parents=True, exist_ok=True)

    # Initialize logger
    logger = Logger(
        log_dir=str(log_dir),
        experiment_name=config.experiment_name
    )

    # Log configuration
    logger.log_config(config.to_dict())

    # Prepare tokenizer
    tokenizer = prepare_tokenizer(config)

    # Prepare dataloaders
    train_dataloader, val_dataloader = prepare_dataloaders(config, tokenizer)

    # Prepare model
    model = prepare_model(config, tokenizer)

    # Calculate training steps
    num_training_steps = config.training.max_steps

    # Prepare optimizer and scheduler
    optimizer, scheduler = prepare_optimizer_and_scheduler(
        config, model, num_training_steps
    )

    # Training loop
    logger.log("Starting training...")
    logger.log(f"Total training steps: {num_training_steps}")
    logger.log(f"Logging every {config.training.logging_steps} steps")
    logger.log(f"Evaluation every {config.training.eval_steps} steps")
    logger.log(f"Saving checkpoints every {config.training.save_steps} steps")
    global_step = 0
    best_val_loss = float('inf')

    train_iter = iter(train_dataloader)

    while global_step < num_training_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            # Reset iterator
            train_iter = iter(train_dataloader)
            batch = next(train_iter)

        # Training step
        metrics = train_step(model, batch, optimizer, scheduler, config)

        global_step += 1

        # Log metrics
        if global_step % config.training.logging_steps == 0:
            progress_pct = (global_step / num_training_steps) * 100
            logger.log(f"Step {global_step}/{num_training_steps} ({progress_pct:.1f}%)")
            logger.log_metrics(global_step, metrics, prefix="train/")

        # Evaluation
        if global_step % config.training.eval_steps == 0:
            logger.log("Running evaluation...")
            eval_metrics = evaluate(model, val_dataloader, config)
            logger.log_metrics(global_step, eval_metrics, prefix="val/")

            # Save best model
            if eval_metrics['loss'] < best_val_loss:
                best_val_loss = eval_metrics['loss']
                logger.log(f"New best validation loss: {best_val_loss:.4f}")
                save_checkpoint(model, optimizer, scheduler, global_step, config, logger, eval_metrics)

        # Regular checkpoint saving
        if global_step % config.training.save_steps == 0:
            save_checkpoint(model, optimizer, scheduler, global_step, config, logger)

    logger.log(f"Training completed! Total steps: {global_step}")

    # Final evaluation
    logger.log("Running final evaluation...")
    final_metrics = evaluate(model, val_dataloader, config)
    logger.log_metrics(global_step, final_metrics, prefix="final/")

    # Save final checkpoint
    save_checkpoint(model, optimizer, scheduler, global_step, config, logger, final_metrics)

    # Save final results
    logger.save_results({
        "best_val_loss": best_val_loss,
        "final_metrics": final_metrics,
        "total_steps": global_step
    })

    logger.close()
    print("\nTraining completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train confidence prediction model")
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON file")
    parser.add_argument("--experiment_name", type=str, default=None, help="Experiment name")
    parser.add_argument("--max_steps", type=int, default=None, help="Maximum training steps")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size per device")
    parser.add_argument("--logging_steps", type=int, default=None, help="Log metrics every N steps")
    parser.add_argument("--eval_steps", type=int, default=None, help="Evaluate every N steps")
    parser.add_argument("--save_steps", type=int, default=None, help="Save checkpoint every N steps")
    parser.add_argument("--max_eval_batches", type=int, default=None, help="Max batches for evaluation (None = all)")

    args = parser.parse_args()

    # Load config
    if args.config:
        config = Config.from_json(args.config)
    else:
        config = get_default_config()

    # Override with command line args
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    if args.max_steps:
        config.training.max_steps = args.max_steps
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.batch_size:
        config.training.per_device_train_batch_size = args.batch_size
    if args.logging_steps:
        config.training.logging_steps = args.logging_steps
    if args.eval_steps:
        config.training.eval_steps = args.eval_steps
    if args.save_steps:
        config.training.save_steps = args.save_steps
    if args.max_eval_batches is not None:
        # Treat 0 as None (evaluate all batches)
        config.eval.max_eval_batches = args.max_eval_batches if args.max_eval_batches > 0 else None

    # Run training
    main(config)
