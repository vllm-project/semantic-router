# Quick Reference Card

## 🚀 One-Line Commands

### Qwen3-0.6B
```bash
python ft_qwen3_generative_lora.py --model "Qwen/Qwen3-0.6B" --mode train --epochs 8 --lora-rank 16 --batch-size 8 --output-dir qwen3_0.6b
```

### Qwen3-1.7B (Single GPU)
```bash
python ft_qwen3_generative_lora.py --model "Qwen/Qwen3-1.7B" --mode train --epochs 5 --lora-rank 16 --lora-dropout 0.1 --batch-size 16 --learning-rate 2e-4 --max-samples-per-category 400 --output-dir qwen3_1.7b
```

### Qwen3-1.7B (8 GPU - Fastest)
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 ft_qwen3_generative_lora.py --model "Qwen/Qwen3-1.7B" --mode train --epochs 5 --lora-rank 16 --lora-dropout 0.1 --batch-size 4 --learning-rate 2e-4 --max-samples-per-category 400 --output-dir qwen3_1.7b_8gpu
```

### Validation
```bash
python ft_qwen3_generative_lora.py --mode validate --model-path <output_dir> --num-val-samples 1578
```

---

## 📊 Model Configurations

| Model | Rank | Alpha | Dropout | LR | Batch | Epochs | Time (1 GPU) | Time (8 GPU) |
|-------|------|-------|---------|-----|-------|--------|--------------|--------------|
| 0.6B | 16 | 32 | 0.05 | 3e-4 | 8 | 8 | ~20 min | ~3 min |
| 1.7B | 16 | 32 | 0.1 | 2e-4 | 16 | 5 | ~10 min | ~5 min |
| 7B+ | 32 | 64 | 0.15 | 1e-4 | 8 | 3-5 | ~60 min | ~10 min |

---

## ⚡ Speed Optimization

| Method | Command | Speedup |
|--------|---------|---------|
| Increase batch size | `--batch-size 32` | 3-4× |
| Use 4 GPUs | `torchrun --nproc_per_node=4` | 4× |
| Use 8 GPUs | `torchrun --nproc_per_node=8` | 7-8× |
| Enable FP16 | Edit line 492: `fp16=True` | +30-50% |

---

## 🚨 Troubleshooting

| Problem | Solution |
|---------|----------|
| Overfitting | `--epochs 3 --lora-dropout 0.15` |
| OOM Error | `--batch-size 2 --lora-rank 8` |
| Low GPU Usage | `--batch-size 32` or use multi-GPU |
| Unstable Training | `--learning-rate 1e-4` |

---

## 📈 Healthy Training Signs

✅ Validation loss decreases or plateaus  
✅ Gradient norm < 1.0  
✅ GPU memory 15-30% utilized  
✅ No NaN values  

❌ Validation loss increases → Stop training!  
❌ Gradient norm > 10 → Reduce learning rate  
❌ GPU memory < 10% → Increase batch size  

---

## 📝 Full Guide

See `TRAINING_GUIDE.md` for detailed explanations and advanced optimizations.

