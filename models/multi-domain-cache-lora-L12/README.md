# Multi-Domain Cache LoRA (L12)

A 4-domain LoRA adapter for semantic cache embeddings, trained on medical, law, programming, and psychology domains.

## Model Details

- **Base Model**: `sentence-transformers/all-MiniLM-L12-v2` (384-dim, 33M params)
- **Adapter Size**: 582 KB
- **LoRA Rank**: 8
- **LoRA Alpha**: 16
- **Target Modules**: q_proj, v_proj
- **Trainable Parameters**: 147,456 (0.44% of base model)

## Training

- **Domains**: Medical, Law, Programming, Psychology
- **Total Triplets**: ~449,000
- **Loss**: Multiple Negatives Ranking (MNR) with temperature=0.05
- **Epochs**: 1
- **Batch Size**: 32
- **Learning Rate**: 1e-4
- **Training Device**: MPS (Apple Silicon)

## Performance

Evaluated on held-out test sets:

| Domain | Baseline Margin | LoRA Margin | Improvement |
|--------|----------------|-------------|-------------|
| Medical | 0.4416 | 0.5517 | **+24.9%** |
| Law | 0.4942 | 0.6270 | **+26.9%** |
| Programming | 0.2415 | 0.2696 | **+11.6%** |
| **Average** | - | - | **+21.1%** |

**Note**: This L12-based model achieves 21.1% average improvement, compared to 26.3% for the L6-based model (all-MiniLM-L6-v2). The performance difference is only -5.2%, making L12 a viable alternative with slightly better base model quality.

## Usage

```python
from sentence_transformers import SentenceTransformer
from peft import PeftModel

# Load base model
base_model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")

# Load LoRA adapter
base_model[0].auto_model = PeftModel.from_pretrained(
    base_model[0].auto_model,
    "models/multi-domain-cache-lora-L12"
)

# Encode queries
embeddings = base_model.encode([
    "What is the treatment for hypertension?",  # Medical
    "Explain contract law principles",          # Law
    "How to implement binary search in Python"  # Programming
])
```

## Files

- `adapter_model.safetensors` (582 KB) - LoRA weights
- `adapter_config.json` - LoRA configuration
- `tokenizer.json` (695 KB) - Tokenizer vocabulary
- `evaluation_results.json` - Detailed evaluation metrics
- `training_info.json` - Training metadata
- `training_history.json` - Training loss history

## Evaluation Methodology

**Margin-based evaluation**:
```
Margin = avg(cosine_sim(anchor, positive)) - avg(cosine_sim(anchor, negative))
```

Higher margins indicate better semantic understanding, where the model correctly identifies semantically similar content as closer and dissimilar content as farther.

## License

Same as base model: Apache 2.0

## Citation

```bibtex
@software{semantic_router_cache_lora,
  title = {Multi-Domain Cache LoRA for Semantic Router},
  author = {Semantic Router Team},
  year = {2025},
  url = {https://github.com/aurelio-labs/semantic-router}
}
```
