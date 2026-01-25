# mmBERT Feedback Detector

Fine-tuning pipeline for training a **4-class user satisfaction classifier** using mmBERT.

Compatible with: [llm-semantic-router/feedback-detector](https://huggingface.co/llm-semantic-router/feedback-detector)

## Key Insight

**Follow-up messages alone contain sufficient signal for classification.** No conversation context needed—just pass the user's response directly.

## Labels

| Label | Description | Example |
|-------|-------------|---------|
| `SAT` | User is satisfied | "Thanks!", "Perfect", "Great!" |
| `NEED_CLARIFICATION` | User needs more explanation | "What do you mean?", "Can you explain?" |
| `WRONG_ANSWER` | System provided incorrect info | "No, that's wrong", "That's not right" |
| `WANT_DIFFERENT` | User wants alternatives | "Show me others", "What else?" |

## Base Model: mmBERT

**mmBERT** (Multilingual ModernBERT) provides:

- **1800+ languages** with 256k vocabulary
- **8192 max context** (RoPE embeddings)
- **Cross-lingual transfer**: train on English, works globally

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

### Train (Full Fine-tuning)

```bash
python train_feedback_detector.py \
    --model_name jhu-clsp/mmBERT-base \
    --output_dir models/mmbert_feedback_detector \
    --epochs 5 \
    --batch_size 16
```

### Train with LoRA

```bash
python train_feedback_detector.py \
    --model_name jhu-clsp/mmBERT-base \
    --output_dir models/mmbert_feedback_detector \
    --use_lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --merge_lora \
    --epochs 5
```

### Inference

```python
from inference_feedback import FeedbackDetector

detector = FeedbackDetector("models/mmbert_feedback_detector")

# Just pass the follow-up message!
result = detector.classify("That's wrong, the answer is 42.")
print(result)  # WRONG_ANSWER (92.3%)

# Batch classification
results = detector.classify_batch([
    "Thanks, that's helpful!",
    "What do you mean?",
    "Show me other options.",
])
```

## Output Models

```
models/
├── mmbert_feedback_detector/           # Full fine-tuned
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer.json
│
├── mmbert_feedback_detector_lora/      # LoRA adapter only
│   ├── adapter_config.json
│   └── adapter_model.safetensors
│
└── mmbert_feedback_detector_merged/    # LoRA merged (for deployment)
    ├── config.json
    ├── model.safetensors
    └── tokenizer.json
```

## Directory Structure

```
modernbert_dissat_pipeline/
├── configs/
│   └── config.py               # Configuration
├── data_processing/            # Dataset processors
├── train_feedback_detector.py  # Training (full + LoRA)
├── inference_feedback.py       # Inference
├── requirements.txt
└── README.md
```

## Integration with Semantic Router

```python
from inference_feedback import FeedbackDetector

detector = FeedbackDetector("models/mmbert_feedback_detector")

def handle_user_response(followup: str, model_used: str):
    result = detector.classify(followup)
    
    if result.label == "WRONG_ANSWER":
        # Strong signal: penalize this model
        apply_heavy_penalty(model_used)
    elif result.label == "NEED_CLARIFICATION":
        # Try more detailed model
        prefer_verbose_model()
    elif result.label == "WANT_DIFFERENT":
        # Try alternative approach
        try_alternative_model()
    else:  # SAT
        # Reinforce this model choice
        reward_model(model_used)
```

## Pre-trained Models

| Model | Description | Link |
|-------|-------------|------|
| **mmbert-feedback-detector-merged** | Ready for inference | [🤗 Hub](https://huggingface.co/llm-semantic-router/mmbert-feedback-detector-merged) |
| **mmbert-feedback-detector-lora** | LoRA adapter | [🤗 Hub](https://huggingface.co/llm-semantic-router/mmbert-feedback-detector-lora) |

### Use Pre-trained Model

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="llm-semantic-router/mmbert-feedback-detector-merged")
result = classifier("Thanks, that's helpful!")
print(result)  # [{'label': 'SAT', 'score': 0.999...}]
```

## References

- [mmBERT Base Model](https://huggingface.co/jhu-clsp/mmBERT-base)
- [feedback-detector-dataset](https://huggingface.co/datasets/llm-semantic-router/feedback-detector-dataset)
- [mmbert-feedback-detector-merged](https://huggingface.co/llm-semantic-router/mmbert-feedback-detector-merged)
- [mmbert-feedback-detector-lora](https://huggingface.co/llm-semantic-router/mmbert-feedback-detector-lora)
