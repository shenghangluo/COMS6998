# Troubleshooting Guide

## Common Issues and Solutions

### Out of Memory (OOM)

**Symptom**: `CUDA out of memory` error during training

**Solutions**:

1. **Use LoRA instead of full fine-tuning**:
   ```bash
   python train_lora.py  # Uses ~50% less memory
   ```

2. **Use LoRA or reduce batch size for full training**:
   ```bash
   python train_lora.py --experiment_name test  # LoRA uses less memory
   python train.py --batch_size 2               # Or reduce batch size for full
   ```

3. **Reduce sequence length**:
   ```python
   config.data.max_length = 1024  # From 2048
   config.data.max_steps_per_example = 5  # From 10
   ```

4. **Increase gradient accumulation**:
   ```python
   config.training.per_device_train_batch_size = 2
   config.training.gradient_accumulation_steps = 8
   # Maintains effective batch size of 16
   ```

5. **Use smaller model** (if applicable):
   ```python
   config.model.mlp_hidden_dims = [512, 128]  # Smaller MLP
   ```

### GPU Not Detected

**Symptom**: Training runs on CPU instead of GPU

**Check GPU availability**:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
```

**Solutions**:

1. Verify CUDA installation:
   ```bash
   nvidia-smi
   nvcc --version
   ```

2. Reinstall PyTorch with CUDA:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```

3. Check driver compatibility

### Slow Training

**Symptom**: Training is slower than expected

**Solutions**:

1. **Use LoRA** (30-40% faster):
   ```bash
   python train_lora.py
   ```

2. **Increase batch size** (if memory allows):
   ```bash
   python train.py --batch_size 8
   ```

3. **Reduce dataloader workers** (if CPU bottleneck):
   ```python
   config.training.dataloader_num_workers = 2
   ```

4. **Use bfloat16** (instead of float32):
   ```python
   config.training.bf16 = True
   config.model.torch_dtype = "bfloat16"
   ```

5. **Limit dataset size** (for testing):
   ```python
   config.data.max_train_samples = 10000
   ```

### Model Not Learning

**Symptom**: Loss not decreasing, metrics not improving

**Diagnose**:
```bash
# Check if loss is changing at all
tail -f outputs/*.log | grep "loss="
```

**Solutions**:

1. **Check learning rate** (may be too low):
   ```bash
   # LoRA
   python train_lora.py --learning_rate 1e-4  # Try 5e-4

   # Full fine-tuning
   python train.py --learning_rate 2e-5  # Try 5e-5
   ```

2. **Verify confidence token is added**:
   ```python
   # Should print token ID
   print(tokenizer.convert_tokens_to_ids("<CONFIDENCE>"))
   ```

3. **Check data loading**:
   ```python
   # Verify labels are in correct range [0, 1]
   from data.dataset import ConfidenceDataset
   # Inspect batch['labels']
   ```

4. **Reduce dropout** (if underfitting):
   ```python
   config.model.mlp_dropout = 0.05  # From 0.1
   ```

5. **Train longer**:
   ```bash
   python train.py --max_steps 20000  # From 10000
   ```

### Loss Increases After Initial Decrease

**Symptom**: Loss goes down, then starts increasing

**Cause**: Overfitting or learning rate too high

**Solutions**:

1. **Lower learning rate**:
   ```bash
   python train_lora.py --learning_rate 5e-5  # From 1e-4
   ```

2. **Increase regularization**:
   ```python
   config.model.mlp_dropout = 0.2  # From 0.1
   config.training.weight_decay = 0.05  # From 0.01
   ```

3. **Use more data**:
   ```python
   config.data.max_train_samples = None  # Use all data
   ```

4. **Early stopping** (manual):
   - Monitor validation loss
   - Stop when it stops improving

### Data Loading Errors

**Symptom**: `FileNotFoundError` or data loading failures

**Solutions**:

1. **Verify dataset paths**:
   ```bash
   ls /workspace/datasets/PRM800K/
   ls /workspace/datasets/reveal/
   ls /workspace/datasets/eQASC/
   ```

2. **Check file names**:
   ```python
   # PRM800K
   ls /workspace/datasets/PRM800K/phase2_train.jsonl
   ls /workspace/datasets/PRM800K/phase2_test.jsonl

   # REVEAL
   ls /workspace/datasets/reveal/eval.jsonl
   ls /workspace/datasets/reveal/open.jsonl

   # eQASC
   ls /workspace/datasets/eQASC/eqasc_train_grc.json
   ```

3. **Update paths in config**:
   ```python
   config.data.prm800k_path = "/correct/path/to/PRM800K"
   ```

### Tokenization Errors

**Symptom**: `KeyError` or token not found

**Solutions**:

1. **Ensure confidence token is added**:
   ```python
   tokenizer.add_special_tokens({'additional_special_tokens': ['<CONFIDENCE>']})
   ```

2. **Resize model embeddings**:
   ```python
   model.backbone.resize_token_embeddings(len(tokenizer))
   ```

3. **Verify token ID is set**:
   ```python
   model.set_confidence_token_id(
       tokenizer.convert_tokens_to_ids('<CONFIDENCE>')
   )
   ```

### Checkpoint Loading Errors

**Symptom**: Cannot load checkpoint

**Solutions**:

1. **Verify checkpoint exists**:
   ```bash
   ls outputs/my_experiment/checkpoint_step_1000/
   ```

2. **Check file structure**:
   ```bash
   # Should contain:
   ls backbone.pt mlp_head.pt  # Full fine-tuning
   # OR
   ls lora_adapter/ mlp_head.pt  # LoRA
   ```

3. **Match model type** (full vs LoRA):
   ```python
   # If trained with LoRA, evaluate with LoRA
   python evaluate.py --checkpoint <path>  # Loads from config.json
   ```

### Evaluation Errors

**Symptom**: Evaluation fails or produces NaN

**Solutions**:

1. **Check for valid predictions**:
   ```python
   # Predictions should be [0, 1]
   # NaN indicates model issue
   ```

2. **Verify test data exists**:
   ```bash
   ls /workspace/datasets/*/test*
   ls /workspace/datasets/*/open*
   ```

3. **Ensure model is in eval mode**:
   ```python
   model.eval()  # Should be automatic
   ```

### Import Errors

**Symptom**: `ModuleNotFoundError`

**Solutions**:

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Check Python path**:
   ```python
   import sys
   print(sys.path)
   # Should include /workspace/Main
   ```

3. **Run from project root**:
   ```bash
   cd /workspace/Main
   python train.py
   ```

### NaN Loss or Predictions

**Symptom**: Loss becomes NaN during training

**Causes**: Gradient explosion, learning rate too high, or numerical instability

**Solutions**:

1. **Lower learning rate**:
   ```bash
   python train.py --learning_rate 1e-5  # From 2e-5
   ```

2. **Check gradient clipping**:
   ```python
   config.training.max_grad_norm = 0.5  # From 1.0
   ```

3. **Use mixed precision carefully**:
   ```python
   config.training.bf16 = True  # bfloat16 more stable than fp16
   config.training.fp16 = False
   ```

4. **Check for invalid labels**:
   ```python
   # Labels should be [0, 1], not NaN or Inf
   ```

## Performance Issues

### Training Slower Than Expected

**Expected performance** (A100 40GB):
- LoRA: ~7-10 samples/sec
- Full: ~5-8 samples/sec

**If slower**:

1. Check GPU utilization:
   ```bash
   nvidia-smi -l 1  # Monitor every second
   ```

2. Increase batch size if GPU not fully utilized

3. Check dataloader bottleneck:
   ```python
   config.training.dataloader_num_workers = 8  # From 4
   config.training.dataloader_pin_memory = True
   ```

### Memory Not Released

**Symptom**: GPU memory not freed after training

**Solutions**:

1. **Explicit cleanup**:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

2. **Restart Python process**

3. **Kill hanging processes**:
   ```bash
   nvidia-smi
   # Note PID of process
   kill -9 <PID>
   ```

## Getting Help

If issues persist:

1. **Check logs**:
   ```bash
   cat outputs/<experiment>*.log
   ```

2. **Enable debug logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

3. **Test with minimal config**:
   ```bash
   python train_lora.py --experiment_name test --max_steps 100
   ```

4. **Verify installation**:
   ```bash
   python -c "import torch, transformers, peft; print('OK')"
   ```

## Debug Checklist

When troubleshooting:

- [ ] Datasets accessible at correct paths
- [ ] GPU detected and CUDA available
- [ ] Dependencies installed (requirements.txt)
- [ ] Confidence token added to tokenizer
- [ ] Model embeddings resized
- [ ] Config matches training mode (full vs LoRA)
- [ ] Sufficient disk space for checkpoints
- [ ] Logs being written to outputs/

## See Also

- **[Training Guide](training.md)** - Training workflow
- **[Configuration Reference](configuration.md)** - Config options
- **[Best Practices](best-practices.md)** - Optimization tips
