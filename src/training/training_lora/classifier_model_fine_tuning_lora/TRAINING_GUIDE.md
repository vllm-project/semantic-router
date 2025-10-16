# Qwen3 Fine-Tuning Guide: Optimized Training & Multi-GPU Acceleration

## 📋 Quick Reference

### Model-Specific Configurations

| Model | LoRA Rank | LoRA Alpha | Dropout | LR | Batch Size | Epochs |
|-------|-----------|------------|---------|-----|------------|--------|
| **Qwen3-0.6B** | 8-16 | 16-32 | 0.05 | 3e-4 | 4-8 | 5-8 |
| **Qwen3-1.7B** | 16-32 | 32-64 | 0.1 | 2e-4 | 8-16 | 3-5 |
| **Qwen3-7B+** | 32-64 | 64-128 | 0.1-0.15 | 1e-4 | 4-8 | 3-5 |

---

## 🚀 Single GPU Training

### Qwen3-0.6B (Default)
```bash
python ft_qwen3_generative_lora.py \
  --model "Qwen/Qwen3-0.6B" \
  --mode train \
  --epochs 8 \
  --lora-rank 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --batch-size 8 \
  --learning-rate 3e-4 \
  --max-samples-per-category 150 \
  --output-dir qwen3_0.6b_lora
```

**Expected:** ~15-20 min on A800 (80GB), ~8GB GPU memory

### Qwen3-1.7B (Optimized)
```bash
python ft_qwen3_generative_lora.py \
  --model "Qwen/Qwen3-1.7B" \
  --mode train \
  --epochs 5 \
  --lora-rank 16 \
  --lora-alpha 32 \
  --lora-dropout 0.1 \
  --batch-size 16 \
  --learning-rate 2e-4 \
  --max-samples-per-category 400 \
  --output-dir qwen3_1.7b_lora
```

**Expected:** ~10-15 min on A800 (80GB), ~25GB GPU memory

---

## ⚡ Multi-GPU Training (Recommended for Speed)

### 8 GPU Training (Fastest)
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --nproc_per_node=8 \
  ft_qwen3_generative_lora.py \
  --model "Qwen/Qwen3-1.7B" \
  --mode train \
  --epochs 5 \
  --lora-rank 16 \
  --lora-alpha 32 \
  --lora-dropout 0.1 \
  --batch-size 4 \
  --learning-rate 2e-4 \
  --max-samples-per-category 400 \
  --output-dir qwen3_1.7b_8gpu
```

**Expected:** ~5 min on 8×A800, ~8GB per GPU, 7-8× speedup

### 4 GPU Training (Balanced)
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node=4 \
  ft_qwen3_generative_lora.py \
  --model "Qwen/Qwen3-1.7B" \
  --mode train \
  --epochs 5 \
  --lora-rank 16 \
  --lora-alpha 32 \
  --lora-dropout 0.1 \
  --batch-size 4 \
  --learning-rate 2e-4 \
  --max-samples-per-category 400 \
  --output-dir qwen3_1.7b_4gpu
```

**Expected:** ~9 min on 4×A800, 4× speedup

---

## 📊 Performance Comparison

| Configuration | GPUs | Batch Size | Time | Speedup | GPU Memory |
|---------------|------|------------|------|---------|------------|
| Original (1.7B) | 1 | 4 | 36 min | 1× | 6.8GB |
| Optimized Single GPU | 1 | 16 | 10 min | 3.6× | 25GB |
| 4 GPU Distributed | 4 | 4×4=16 | 9 min | 4× | 8GB/GPU |
| **8 GPU Distributed** | **8** | **4×8=32** | **5 min** | **7-8×** | **8GB/GPU** |

---

## 🔧 Validation

### Single GPU Validation
```bash
python ft_qwen3_generative_lora.py \
  --mode validate \
  --model-path qwen3_1.7b_lora \
  --num-val-samples 1578
```

### Multi-GPU Validation (Faster)
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --nproc_per_node=8 \
  ft_qwen3_generative_lora.py \
  --mode validate \
  --model-path qwen3_1.7b_8gpu \
  --num-val-samples 1578
```

**Validation speedup:** 10 min → 1-2 min with 8 GPUs

---

## 🎯 Hyperparameter Guidelines

### LoRA Configuration

**LoRA Rank:**
- Too low (4-8): Underfitting, poor performance
- Optimal (16-32 for 1.7B): Best balance
- Too high (64+): Overfitting, slower training

**LoRA Alpha:**
- Rule of thumb: `alpha = 2 × rank`
- Example: rank=16 → alpha=32

**LoRA Dropout:**
- Small models (0.6B): 0.05
- Medium models (1.7B): 0.1
- Large models (7B+): 0.1-0.15
- Large datasets: Increase dropout to prevent overfitting

### Learning Rate

| Model Size | Learning Rate | Notes |
|------------|---------------|-------|
| 0.6B | 3e-4 | Higher LR for small models |
| 1.7B | 2e-4 | Balanced |
| 7B+ | 1e-4 | Lower LR for stability |

### Batch Size

**Single GPU:**
- 16GB GPU: batch_size=2-4
- 40GB GPU: batch_size=8-16
- 80GB GPU: batch_size=16-32

**Multi-GPU:**
- Per-GPU batch size: 4-8
- Total batch size = per_gpu_batch × num_gpus
- Recommended total: 16-32

### Epochs

**Guidelines:**
- Start with 3-5 epochs
- Monitor validation loss
- Stop if validation loss increases (overfitting)
- Use early stopping (enabled by default)

---

## 🚨 Common Issues & Solutions

### Issue 1: Overfitting
**Symptoms:**
- Training loss decreases, validation loss increases
- 100% accuracy on small validation set

**Solutions:**
```bash
# Reduce epochs
--epochs 3

# Increase dropout
--lora-dropout 0.15

# Reduce samples
--max-samples-per-category 300

# Increase weight decay (modify script line 481)
weight_decay=0.1
```

### Issue 2: Low GPU Utilization
**Symptoms:**
- GPU memory < 20% on 80GB GPU
- Training slower than expected

**Solutions:**
```bash
# Increase batch size
--batch-size 32

# Use multi-GPU
torchrun --nproc_per_node=8 ...

# Enable FP16 (modify script line 492)
fp16=True
```

### Issue 3: Out of Memory (OOM)
**Solutions:**
```bash
# Reduce batch size
--batch-size 2

# Enable gradient checkpointing (modify script line 493)
gradient_checkpointing=True

# Reduce LoRA rank
--lora-rank 8
```

### Issue 4: Unstable Training
**Symptoms:**
- Loss spikes or NaN values
- Gradient norm > 10

**Solutions:**
```bash
# Reduce learning rate
--learning-rate 1e-4

# Increase warmup (modify script line 490)
warmup_ratio=0.15

# Check gradient clipping (default: max_grad_norm=1.0)
```

---

## 📈 Monitoring Training

### Watch for Healthy Training Patterns

✅ **Good Training:**
```
Epoch 1: train_loss=2.1, val_loss=0.80, grad_norm=0.15
Epoch 2: train_loss=1.5, val_loss=0.76, grad_norm=0.13
Epoch 3: train_loss=1.2, val_loss=0.74, grad_norm=0.14
Epoch 4: train_loss=1.0, val_loss=0.73, grad_norm=0.12
Epoch 5: train_loss=0.9, val_loss=0.73, grad_norm=0.11  ← Plateau OK
```

❌ **Overfitting:**
```
Epoch 1: train_loss=2.1, val_loss=0.80
Epoch 2: train_loss=1.5, val_loss=0.76
Epoch 3: train_loss=1.2, val_loss=0.74
Epoch 4: train_loss=0.8, val_loss=0.78  ← Val loss increasing!
Epoch 5: train_loss=0.5, val_loss=0.85  ← Stop training!
```

### Monitor GPU Usage
```bash
# Terminal 1: Run training
torchrun --nproc_per_node=8 ft_qwen3_generative_lora.py ...

# Terminal 2: Monitor GPUs
watch -n 1 nvidia-smi
```

**Expected GPU utilization:**
- Single GPU: 15-30% of total memory
- Multi-GPU: All GPUs should show active processes

---

## 🔬 Advanced Optimizations

### Enable FP16 Mixed Precision
Edit `ft_qwen3_generative_lora.py` line 492:
```python
fp16=True,  # Enable for 30-50% speedup
```

**Benefits:**
- 30-50% faster training
- 40% less memory usage
- Supported on A800/A100/V100 GPUs

### Enable Gradient Checkpointing
Edit `ft_qwen3_generative_lora.py` line 493:
```python
gradient_checkpointing=True,  # Enable for larger batch sizes
```

**Benefits:**
- Allows 2× larger batch sizes
- ~10% slower but worth it for memory savings

### Increase Weight Decay
Edit `ft_qwen3_generative_lora.py` line 481:
```python
weight_decay=0.1,  # Increase for stronger regularization
```

---

## 📝 Best Practices

### 1. Start Conservative
```bash
# First run: Conservative settings
python ft_qwen3_generative_lora.py \
  --epochs 3 \
  --lora-rank 8 \
  --lora-dropout 0.15 \
  --batch-size 4 \
  --learning-rate 1e-4 \
  --max-samples-per-category 200
```

### 2. Monitor & Adjust
- Watch validation loss closely
- Stop if validation loss increases
- Adjust hyperparameters based on results

### 3. Validate on Full Dataset
```bash
# Don't trust small validation samples
python ft_qwen3_generative_lora.py \
  --mode validate \
  --model-path <your_model> \
  --num-val-samples 1578  # Full validation set
```

### 4. Use Multi-GPU for Production
```bash
# Production training: Use all available GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --nproc_per_node=8 \
  ft_qwen3_generative_lora.py \
  --model "Qwen/Qwen3-1.7B" \
  --epochs 5 \
  --lora-rank 16 \
  --lora-alpha 32 \
  --lora-dropout 0.1 \
  --batch-size 4 \
  --learning-rate 2e-4 \
  --max-samples-per-category 400 \
  --output-dir qwen3_1.7b_production
```

---

## 🎓 Summary

### Quick Start Commands

**Qwen3-0.6B (Single GPU):**
```bash
python ft_qwen3_generative_lora.py --model "Qwen/Qwen3-0.6B" --mode train --epochs 8 --lora-rank 16 --batch-size 8 --output-dir qwen3_0.6b
```

**Qwen3-1.7B (Single GPU, Optimized):**
```bash
python ft_qwen3_generative_lora.py --model "Qwen/Qwen3-1.7B" --mode train --epochs 5 --lora-rank 16 --lora-dropout 0.1 --batch-size 16 --learning-rate 2e-4 --max-samples-per-category 400 --output-dir qwen3_1.7b
```

**Qwen3-1.7B (8 GPU, Fastest):**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 ft_qwen3_generative_lora.py --model "Qwen/Qwen3-1.7B" --mode train --epochs 5 --lora-rank 16 --lora-dropout 0.1 --batch-size 4 --learning-rate 2e-4 --max-samples-per-category 400 --output-dir qwen3_1.7b_8gpu
```

### Key Takeaways

1. **Model-specific tuning matters:** 1.7B needs different hyperparameters than 0.6B
2. **Multi-GPU = 7-8× speedup:** Use `torchrun` for distributed training
3. **Watch for overfitting:** Stop when validation loss increases
4. **Batch size matters:** Larger batches = faster training (if GPU memory allows)
5. **Validate properly:** Use full validation set, not just 20 samples

---

## 📚 Additional Resources

- Script documentation: See docstring in `ft_qwen3_generative_lora.py`
- Qwen3 model cards: https://huggingface.co/Qwen
- LoRA paper: https://arxiv.org/abs/2106.09685
- PyTorch distributed training: https://pytorch.org/tutorials/beginner/dist_overview.html

