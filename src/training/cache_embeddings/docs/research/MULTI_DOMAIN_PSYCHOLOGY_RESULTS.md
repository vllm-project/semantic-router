# Multi-Domain + Psychology LoRA Results

**Date**: 2026-01-15
**Model**: Multi-domain cache embedding LoRA (Medical + Law + Programming + Psychology)
**Training Data**: 509,752 triplets (1 epoch)
**Base Model**: sentence-transformers/all-MiniLM-L6-v2
**Evaluation**: 1000-sample random subset per domain

---

## Executive Summary

The merged multi-domain LoRA model with psychology shows **strong improvements across all 4 domains**, with psychology achieving the highest gain (+34.9%). All domains show double-digit improvements, validating the multi-domain training approach.

**Average improvement: +19.4%** (across 4 domains)

---

## Domain-by-Domain Results

### 1. Psychology (NEW) - +34.9% 🏆

```
Baseline margin: 0.2690
LoRA margin:     0.3628
Absolute gain:   +0.0938
Improvement:     +34.9%
```

**Best performer!** The newly added psychology domain shows the strongest improvement, benefiting from:

- Fixed NUCLEAR strategy (related topic, different aspect)
- 60,656 high-quality triplets generated with Qwen 7B
- Clean training data from Reddit psychology discussions

**Similarity breakdown**:

- Positive sim: 0.5981 → 0.6565 (+9.8%)
- Negative sim: 0.3290 → 0.2937 (-10.7%)

### 2. Law - +16.9%

```
Baseline margin: 0.5505
LoRA margin:     0.6433
Absolute gain:   +0.0928
Improvement:     +16.9%
```

Strong performance, pushing negatives far apart (0.1528 → 0.0628, -58.9% reduction).

### 3. Medical - +14.6%

```
Baseline margin: 0.6494
LoRA margin:     0.7440
Absolute gain:   +0.0945
Improvement:     +14.6%
```

Solid improvement with highest absolute margin (0.7440), showing excellent separation.

### 4. Programming - +11.3%

```
Baseline margin: 0.2415
LoRA margin:     0.2687
Absolute gain:   +0.0272
Improvement:     +11.3%
```

Modest but positive improvement. Programming has inherently lower margins due to technical overlap.

---

## Performance Comparison

### Multi-Domain vs Single-Domain LoRAs

| Domain | Multi-Domain (4-way) | Single-Domain | Difference |
|--------|---------------------|---------------|------------|
| Medical | +14.6% | +35.6% | -21.0% |
| Law | +16.9% | +18.6% | -1.7% |
| Programming | +11.3% | +24.8% | -13.5% |
| Psychology | +34.9% | +37.7% (1-epoch) | -2.8% |
| **Average** | **+19.4%** | **+29.2%** | **-9.8%** |

**Key findings**:

1. **Law shows minimal degradation** (-1.7%) - multi-domain helps law almost as much as single-domain!
2. **Psychology maintains strong performance** (-2.8%) - excellent transfer learning
3. **Medical takes biggest hit** (-21.0%) - may need domain-specific tuning
4. **Trade-off**: ~10% average loss for 4× domain coverage

### Single Model vs Multiple Models

**Multi-domain advantage**:

- **1 model** covers 4 domains (simpler deployment)
- **75% fewer parameters** to manage (1 vs 4 LoRA adapters)
- **Average +19.4%** improvement is still strong
- Law and psychology maintain near-single-domain performance

**Single-domain advantage**:

- **Higher peak performance** per domain
- Best for mission-critical single-domain applications
- Medical gains most from dedicated model

---

## Training Details

### Dataset Composition

| Domain | Triplets | Percentage |
|--------|----------|-----------|
| Medical | 71,200 | 14.0% |
| Law | 229,440 | 45.0% |
| Programming | 148,456 | 29.1% |
| Psychology | 60,656 | 11.9% |
| **Total** | **509,752** | **100%** |

### Training Configuration

- **Epochs**: 1 (for all domains)
- **Batch size**: 32
- **Learning rate**: 2e-4
- **LoRA rank**: 8
- **LoRA alpha**: 16
- **Trainable params**: 73,728 (0.32% of total)
- **Loss**: MultipleNegativesRankingLoss (temperature=0.05)

### Training Speed Comparison

| Environment | Speed | Total Time |
|-------------|-------|-----------|
| Local (M4 Pro, 20 GPU cores) | 13.3 it/s | ~20 min |
| AWS (A10G GPU × 1) | 9.6 it/s | ~27 min |

**Winner**: M4 Pro is 38% faster! (Single-GPU AWS training)

---

## Margin Analysis

### Absolute Margins (Higher is Better)

```
Medical:      0.7440  ████████████████████████████████████████ (highest)
Law:          0.6433  ███████████████████████████████████
Psychology:   0.3628  ████████████████████
Programming:  0.2687  ██████████████
```

**Medical and law** have the highest margins, indicating strong separation between positive and negative pairs.

### Relative Improvements (vs Baseline)

```
Psychology:   +34.9%  █████████████████████████████████████████ (best)
Law:          +16.9%  ████████████████████
Medical:      +14.6%  ████████████████
Programming:  +11.3%  ████████████
```

**Psychology** shows the strongest relative improvement despite starting from a lower baseline.

---

## Recommendations

### For Production Deployment

**Use multi-domain LoRA if**:

- You need coverage across multiple domains
- Deployment simplicity matters (1 model vs 4)
- +19.4% average improvement is sufficient
- Law or psychology are primary use cases

**Use single-domain LoRAs if**:

- Maximum performance in one domain is critical
- Medical domain is the primary focus (+35.6% vs +14.6%)
- Resources allow managing multiple models

### For Future Improvements

1. **Investigate medical domain loss**: Why does medical drop 21% in multi-domain?
   - May need domain-specific fine-tuning
   - Could try domain-adaptive LoRA (separate adapters per domain)

2. **Optimize training data mix**: Current distribution is uneven (law 45%, psychology 12%)
   - Try balanced sampling (25% each)
   - Or weighted loss by domain

3. **Psychology domain**: Consider generating more data
   - Current 60K triplets (12% of total)
   - Could expand to 100K+ for better representation

---

## Files Generated

### Models

- **Multi-domain LoRA**: `/Users/yovadia/cache_embedding_triplets/multi-domain/multi-domain-with-psychology-lora/`
  - adapter_model.safetensors (291KB)
  - adapter_config.json
  - Training history and info

### Data

- **Training data**: `multi-domain-with-psychology.jsonl` (509,752 triplets, 191MB)
- **Evaluation results**:
  - eval_medical.json
  - eval_law.json
  - eval_programming.json
  - eval_psychology.json

### Documentation

- [PSYCHOLOGY_EPOCH_COMPARISON.md](PSYCHOLOGY_EPOCH_COMPARISON.md) - 1-epoch vs 3-epoch analysis
- [PSYCHOLOGY_NUCLEAR_FIX.md](PSYCHOLOGY_NUCLEAR_FIX.md) - NUCLEAR strategy fix details
- [PARALLEL_TRAINING_COMPARISON.md](PARALLEL_TRAINING_COMPARISON.md) - M4 Pro vs AWS GPU comparison

---

## Conclusion

The multi-domain + psychology LoRA is a **successful integration**, showing:

- ✅ **Strong overall performance** (+19.4% average)
- ✅ **Psychology ranks #1** in relative improvement (+34.9%)
- ✅ **Law maintains near-single-domain performance** (-1.7%)
- ✅ **Simplified deployment** (1 model for 4 domains)
- ⚠️ **Medical domain trade-off** (-21% vs single-domain)

**Next steps**: Consider domain-adaptive LoRA architecture for better medical performance while maintaining multi-domain benefits.
