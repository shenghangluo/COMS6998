"""Training script for confidence prediction with LoRA (parameter-efficient fine-tuning)."""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_scheduler
from tqdm import tqdm
import argparse
import bitsandbytes as bnb

# Add LLM directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models.lora_model import ConfidenceModelLoRA
from data.dataset import ConfidenceDataset, collate_fn
from utils.metrics import compute_all_metrics
from utils.logger import Logger
from config import Config, get_lora_config


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_tokenizer(config: Config):
    """Prepare tokenizer with confidence token."""
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
    """Prepare train and validation dataloaders."""
    print("\nPreparing datasets...")

    train_dataset_configs = config.get_dataset_configs("train")
    val_dataset_configs = config.get_dataset_configs("test")

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
    """Prepare LoRA model with MLP head."""
    print("\nInitializing LoRA model...")

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }
    torch_dtype = dtype_map.get(config.model.torch_dtype, torch.bfloat16)

    # Create LoRA model
    model = ConfidenceModelLoRA(
        model_name=config.model.model_name,
        lora_config={
            "r": config.lora.lora_r,
            "lora_alpha": config.lora.lora_alpha,
            "lora_dropout": config.lora.lora_dropout,
            "target_modules": config.lora.lora_target_modules
        },
        mlp_config={
            "hidden_dims": config.model.mlp_hidden_dims,
            "dropout": config.model.mlp_dropout,
            "activation": config.model.mlp_activation
        },
        device_map=config.model.device_map,
        torch_dtype=torch_dtype
    )

    # Resize token embeddings
    model.backbone.resize_token_embeddings(len(tokenizer))

    # Enable gradient checkpointing to reduce memory usage
    if hasattr(model.backbone, 'gradient_checkpointing_enable'):
        model.backbone.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

    # Set confidence token ID
    confidence_token_id = tokenizer.convert_tokens_to_ids(config.model.confidence_token)
    model.set_confidence_token_id(confidence_token_id)

    # Print trainable parameters (LoRA specific)
    print("\nTrainable parameters breakdown:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")

    return model


def prepare_optimizer_and_scheduler(config: Config, model, num_training_steps: int):
    """Prepare optimizer and scheduler for LoRA training."""
    # Only optimize trainable parameters (LoRA + MLP head)
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))

    # Use 8-bit AdamW optimizer to save memory
    # This stores optimizer states in 8-bit format
    optimizer = bnb.optim.AdamW8bit(
        trainable_params,
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
    print("Optimizer states use 8-bit precision (saves memory)")

    return optimizer, scheduler


def train_step(model, batch, optimizer, scheduler, config):
    """Perform a single training step."""
    model.train()

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    confidence_positions = batch['confidence_positions'].to(device)
    labels = batch['labels'].to(device=device, dtype=dtype)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        confidence_positions=confidence_positions,
        labels=labels,
        return_dict=True
    )

    loss = outputs['loss']
    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        filter(lambda p: p.requires_grad, model.parameters()),
        config.training.max_grad_norm
    )

    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    return {
        'loss': loss.item(),
        'learning_rate': scheduler.get_last_lr()[0]
    }


@torch.no_grad()
def evaluate(model, dataloader, config):
    """Evaluate model on validation set."""
    model.eval()
    device = next(model.parameters()).device

    all_predictions = []
    all_labels = []
    all_sources = []
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Evaluating"):
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

        for i in range(predictions.size(0)):
            valid_mask = labels[i] >= 0
            if valid_mask.any():
                all_predictions.append(predictions[i][valid_mask].float().cpu().numpy())
                all_labels.append(labels[i][valid_mask].float().cpu().numpy())
                all_sources.append(batch['source'][i])

        if outputs['loss'] is not None:
            total_loss += outputs['loss'].item()
            num_batches += 1

    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)

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
    """Save LoRA model checkpoint."""
    checkpoint_dir = Path(config.output_dir) / config.experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f"checkpoint_step_{step}"
    checkpoint_path.mkdir(exist_ok=True)

    # Save LoRA adapter
    model.save_lora_adapter(str(checkpoint_path / "lora_adapter"))

    # Save MLP head
    model.save_mlp_head(str(checkpoint_path / "mlp_head.pt"))

    # Save optimizer and scheduler
    torch.save(optimizer.state_dict(), checkpoint_path / "optimizer.pt")
    torch.save(scheduler.state_dict(), checkpoint_path / "scheduler.pt")

    # Save config
    config.save(str(checkpoint_path / "config.json"))

    logger.save_checkpoint_info(step, str(checkpoint_path), metrics or {})

    print(f"LoRA checkpoint saved to {checkpoint_path}")

    # Clean up old checkpoints, keeping only the latest 3
    _cleanup_old_checkpoints(checkpoint_dir, keep_last=3)


def main(config: Config):
    """Main training function for LoRA."""
    set_seed(config.training.seed)

    logger = Logger(
        log_dir=config.log_dir,
        experiment_name=config.experiment_name
    )

    logger.log_config(config.to_dict())

    tokenizer = prepare_tokenizer(config)
    train_dataloader, val_dataloader = prepare_dataloaders(config, tokenizer)
    model = prepare_model(config, tokenizer)

    num_training_steps = config.training.max_steps
    optimizer, scheduler = prepare_optimizer_and_scheduler(config, model, num_training_steps)

    logger.log("Starting LoRA training...")
    global_step = 0
    best_val_loss = float('inf')

    progress_bar = tqdm(total=num_training_steps, desc="Training (LoRA)")

    train_iter = iter(train_dataloader)

    while global_step < num_training_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataloader)
            batch = next(train_iter)

        metrics = train_step(model, batch, optimizer, scheduler, config)

        global_step += 1
        progress_bar.update(1)

        if global_step % config.training.logging_steps == 0:
            logger.log_metrics(global_step, metrics, prefix="train/")

        if global_step % config.training.eval_steps == 0:
            logger.log("Running evaluation...")
            eval_metrics = evaluate(model, val_dataloader, config)
            logger.log_metrics(global_step, eval_metrics, prefix="val/")

            if eval_metrics['loss'] < best_val_loss:
                best_val_loss = eval_metrics['loss']
                logger.log(f"New best validation loss: {best_val_loss:.4f}")
                save_checkpoint(model, optimizer, scheduler, global_step, config, logger, eval_metrics)

        if global_step % config.training.save_steps == 0:
            save_checkpoint(model, optimizer, scheduler, global_step, config, logger)

    progress_bar.close()

    logger.log("Running final evaluation...")
    final_metrics = evaluate(model, val_dataloader, config)
    logger.log_metrics(global_step, final_metrics, prefix="final/")

    save_checkpoint(model, optimizer, scheduler, global_step, config, logger, final_metrics)

    logger.save_results({
        "best_val_loss": best_val_loss,
        "final_metrics": final_metrics,
        "total_steps": global_step
    })

    logger.close()
    print("\nLoRA training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train confidence prediction model with LoRA")
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON file")
    parser.add_argument("--experiment_name", type=str, default=None, help="Experiment name")
    parser.add_argument("--max_steps", type=int, default=None, help="Maximum training steps")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=None, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=None, help="LoRA alpha")

    args = parser.parse_args()

    if args.config:
        config = Config.from_json(args.config)
    else:
        config = get_lora_config()

    if args.experiment_name:
        config.experiment_name = args.experiment_name
    if args.max_steps:
        config.training.max_steps = args.max_steps
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.lora_r:
        config.lora.lora_r = args.lora_r
    if args.lora_alpha:
        config.lora.lora_alpha = args.lora_alpha

    main(config)
