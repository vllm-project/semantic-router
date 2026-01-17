# Psychology LoRA: 1-Epoch vs 3-Epoch Comparison

**Date**: 2026-01-15
**Domain**: Psychology
**Training Data**: 60,656 triplets from 33,809 queries (1.79× augmentation)
**Base Model**: sentence-transformers/all-MiniLM-L6-v2
**Evaluation**: 1000-sample random subset

---

## Summary

Both 1-epoch and 3-epoch models show excellent improvements over baseline:

- **1-epoch**: +37.7% margin improvement
- **3-epoch**: +39.8% margin improvement
- **Difference**: Only 2.1% additional gain from 3× more training time

**Recommendation**: Use 1-epoch model for consistency with multi-domain LoRA approach and marginal diminishing returns.

---

## Detailed Results

### Baseline (No LoRA)

```
Avg positive similarity: 0.5981
Avg negative similarity: 0.3290
Margin:                  0.2690
```

### 1-Epoch LoRA

```
Avg positive similarity: 0.6203
Avg negative similarity: 0.2500
Margin:                  0.3704

Improvement: +0.1013 (+37.7%)
```

### 3-Epoch LoRA

```
Avg positive similarity: 0.6167
Avg negative similarity: 0.2407
Margin:                  0.3760

Improvement: +0.1070 (+39.8%)
```

---

## Analysis

### Training Efficiency

| Model | Epochs | Margin | Improvement | Training Time |
|-------|--------|--------|-------------|---------------|
| Baseline | 0 | 0.2690 | - | - |
| 1-epoch | 1 | 0.3704 | +37.7% | ~2.5 min |
| 3-epoch | 3 | 0.3760 | +39.8% | ~7.5 min |

**Marginal gain per epoch**:

- Epoch 1: +37.7% improvement
- Epochs 2-3: +2.1% additional improvement (only 5.6% of total gain)

### Similarity Breakdown

**Positive similarities** (higher is better):

- Baseline: 0.5981
- 1-epoch: 0.6203 (+0.0222, +3.7%)
- 3-epoch: 0.6167 (+0.0186, +3.1%)

**Negative similarities** (lower is better):

- Baseline: 0.3290
- 1-epoch: 0.2500 (-0.0790, -24.0%)
- 3-epoch: 0.2407 (-0.0883, -26.8%)

**Key insight**: Both models improve primarily by **reducing negative similarity** (pushing negatives farther away). 3-epoch model pushes negatives slightly farther (-26.8% vs -24.0%), but at the cost of slightly lower positive similarity.

---

## Comparison with Other Domains

### Multi-Domain LoRA (1 epoch each)

```
Medical:      +35.6% improvement (0.2743 → 0.3720)
Law:          +18.6% improvement (0.3029 → 0.3593)
Programming:  +24.8% improvement (0.3205 → 0.4000)
Average:      +26.3% improvement
```

### Psychology LoRA (1 epoch)

```
Psychology:   +37.7% improvement (0.2690 → 0.3704)
```

**Psychology ranks #1** among all single-domain models tested!

---

## Recommendation

**Use 1-epoch model** for the following reasons:

1. **Consistency**: Matches multi-domain LoRA approach (1 epoch per domain)
2. **Efficiency**: 97.6% of the 3-epoch performance in 33% of the training time
3. **Diminishing returns**: 3× training time yields only 2.1% additional gain
4. **Top performer**: +37.7% beats all other domains (medical +35.6%, programming +24.8%, law +18.6%)

---

## Files Generated

- **1-epoch model**: `~/cache_embedding_triplets/psychology/psychology-cache-lora-1epoch/`
- **3-epoch model**: `~/cache_embedding_triplets/psychology/psychology-cache-lora/`
- **1-epoch eval**: `~/cache_embedding_triplets/psychology/eval_1epoch.json`
- **3-epoch eval**: `~/cache_embedding_triplets/psychology/eval_3epoch.json`

---

## Next Steps

1. Use 1-epoch psychology LoRA as the official psychology cache embedding model
2. Consider merging with multi-domain LoRA for comprehensive coverage
3. Archive 3-epoch model as experimental result
4. Update model registry with psychology-cache-lora-1epoch

---

## Training Configuration

**LoRA Parameters** (both models):

- Rank: 8
- Alpha: 16
- Dropout: 0.1
- Target modules: Linear layers in sentence transformer
- Trainable params: 73,728 (0.32% of total)

**Training Hyperparameters**:

- Learning rate: 2e-4
- Batch size: 32
- Loss: MultipleNegativesRankingLoss (temperature=0.05)
- Optimizer: AdamW

**Data**:

- Training triplets: 60,656
- Source queries: 33,809
- Augmentation: 1.79× (lower than 5× due to raw Reddit text)
- Generation model: Qwen2.5-7B-Instruct
- NUCLEAR strategy: Related topic, different aspect
