#!/bin/bash

# Full Training Entry Point for Confidence Prediction Model
# This script starts full fine-tuning with optimized settings

set -e  # Exit on error

echo "================================"
echo "Full Training - Confidence Prediction"
echo "================================"
echo ""

# Fix CUDA device visibility
unset NVIDIA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0

# Disable Python bytecode cache
export PYTHONDONTWRITEBYTECODE=1

# Set Hugging Face cache directory
export HF_HOME=/workspace/.hf_cache
mkdir -p "${HF_HOME}"

# Change to Main directory
cd /workspace/Main

# Default training configuration
# Override these by passing arguments: ./train.sh [EXPERIMENT_NAME] [MAX_STEPS] [LEARNING_RATE]
EXPERIMENT_NAME="${1:-full_training_$(date +%Y%m%d_%H%M%S)}"
MAX_STEPS=1000000
LEARNING_RATE=1e-5
BATCH_SIZE=32  # Per-device batch size (uses gradient accumulation)
LOGGING_STEPS=100
EVAL_STEPS=1000
SAVE_STEPS=1000
MAX_EVAL_BATCHES=1000  # Max batches for validation (set to 0 for all batches)

echo "Configuration:"
echo "  Experiment: $EXPERIMENT_NAME"
echo "  Max steps: $MAX_STEPS"
echo "  Learning rate: $LEARNING_RATE"
echo "  Batch size (per device): $BATCH_SIZE"
echo "  Logging steps: $LOGGING_STEPS"
echo "  Eval steps: $EVAL_STEPS"
echo "  Save steps: $SAVE_STEPS"
echo "  Max eval batches: $MAX_EVAL_BATCHES"
echo ""
echo "Starting full fine-tuning..."
echo ""

# Run training
python3 train.py \
    --experiment_name "$EXPERIMENT_NAME" \
    --max_steps $MAX_STEPS \
    --learning_rate $LEARNING_RATE \
    --batch_size $BATCH_SIZE \
    --logging_steps $LOGGING_STEPS \
    --eval_steps $EVAL_STEPS \
    --save_steps $SAVE_STEPS \
    --max_eval_batches $MAX_EVAL_BATCHES

# Check training result
if [ $? -eq 0 ]; then
    echo ""
    echo "================================"
    echo "Training completed successfully! ✓"
    echo "================================"
    echo ""
    echo "Outputs: /workspace/outputs/$EXPERIMENT_NAME/"
    echo "  - Checkpoints: /workspace/outputs/$EXPERIMENT_NAME/checkpoints/"
    echo "  - Logs: /workspace/outputs/$EXPERIMENT_NAME/logs/"
    echo ""
    echo "To evaluate the final checkpoint:"
    echo "  python3 evaluate.py \\"
    echo "    --checkpoint /workspace/outputs/$EXPERIMENT_NAME/checkpoints/checkpoint_step_$MAX_STEPS"
    echo ""
else
    echo ""
    echo "================================"
    echo "Training failed! ✗"
    echo "================================"
    echo "Check logs in /workspace/outputs/$EXPERIMENT_NAME/logs/ for details"
    exit 1
fi
