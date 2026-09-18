"""Curiosity exploration: fine-tune LiquidAI/LFM2.5-Encoder-350M (custom
Lfm2BidirectionalModel, interleaved gated-conv + grouped-query-attention blocks)
on the modality-routing task via LoRA.

Key gotcha (verified by direct reproduction, matching the trap @adaamko flagged
in DECISION_RECORD.md): the model card's documented
`AutoModel.from_pretrained(repo, trust_remote_code=True)` path loads a
COMPLETELY RANDOMLY-INITIALIZED model for this checkpoint -- its real weights
are stored under an `lfm2.` prefix that `Lfm2BidirectionalModel`'s own module
structure doesn't expect. The fix: load via
`AutoModelForMaskedLM.from_pretrained(...).lfm2` instead, which loads the
pretrained weights correctly (verified: zero MISSING/UNEXPECTED keys).

No AutoModelForSequenceClassification variant exists for this architecture, so
this wraps the bare encoder body with a masked-mean-pool + linear head, mirroring
the FocalLoss/class-weight recipe used elsewhere in this pipeline for consistency.
"""
import json
import os
import sys

import torch
import torch.nn as nn
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForMaskedLM, AutoTokenizer, Trainer, TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput

from pathlib import Path as _Path

_HERE = _Path(__file__).resolve().parent
_CLASSIFIER_DIR = str(_HERE.parent)
sys.path.append(_CLASSIFIER_DIR)
from modality_routing_bert_finetuning_lora import FocalLoss, MODALITY_LABELS  # noqa: E402

MODEL_ID = "LiquidAI/LFM2.5-Encoder-350M"
DATA_DIR = f"{_CLASSIFIER_DIR}/exported_modality_routing_dataset"
OUTPUT_DIR = os.environ.get("LFM25_OUTPUT_DIR", str(_HERE / "runs" / "lfm25_encoder_finetuned"))
LABEL_TO_ID = {label: idx for idx, label in enumerate(MODALITY_LABELS)}

# LoRA target modules, from direct introspection of named_modules() on the real
# (correctly-loaded) body -- this architecture interleaves two block types:
#   - conv blocks: conv.in_proj / conv.out_proj (short-conv, linear projections)
#   - attention blocks (layers 2,5,8,10,12,14 only): self_attn.q_proj/k_proj/v_proj/out_proj
# plus feed_forward.w1/w2/w3 (gated SwiGLU-style MLP) present on every layer.
LORA_TARGET_MODULES = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.out_proj",
    "feed_forward.w1",
    "feed_forward.w2",
    "feed_forward.w3",
    "conv.in_proj",
    "conv.out_proj",
]


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


class Lfm2ForModalityClassification(nn.Module):
    def __init__(self, body, num_labels, class_weights=None, focal_gamma=2.0, dropout=0.1):
        super().__init__()
        self.body = body
        hidden_size = body.config.hidden_size if hasattr(body, "config") else body.base_model.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels
        self.focal_loss = FocalLoss(alpha=class_weights, gamma=focal_gamma, reduction="mean")

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.body(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        logits = self.classifier(self.dropout(pooled))
        loss = None
        if labels is not None:
            loss = self.focal_loss(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)


def compute_class_weights(train_data, num_classes=3):
    counts = {}
    for item in train_data:
        counts[item["label"]] = counts.get(item["label"], 0) + 1
    total = len(train_data)
    weights = [max(0.5, min((total / (num_classes * counts.get(i, 1))) ** 0.5, 3.0)) for i in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float32)


def tokenize(rows, tokenizer, max_length=256):
    texts = [r["text"] for r in rows]
    labels = [r["label"] for r in rows]
    enc = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    return Dataset.from_dict(
        {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels}
    )


def main():
    print("Loading tokenizer + body (via AutoModelForMaskedLM, NOT the model card's documented AutoModel path)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    mlm_model = AutoModelForMaskedLM.from_pretrained(MODEL_ID, trust_remote_code=True)
    body = mlm_model.lfm2

    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    body = get_peft_model(body, peft_config)
    body.print_trainable_parameters()

    train_rows = load_jsonl(f"{DATA_DIR}/train.jsonl")
    val_rows = load_jsonl(f"{DATA_DIR}/validation.jsonl")
    print(f"Train: {len(train_rows)}, Val: {len(val_rows)}")

    class_weights = compute_class_weights(train_rows)
    print(f"Class weights: {class_weights.tolist()}")

    model = Lfm2ForModalityClassification(body, num_labels=len(MODALITY_LABELS), class_weights=class_weights)

    train_dataset = tokenize(train_rows, tokenizer)
    val_dataset = tokenize(val_rows, tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=3e-5,
        weight_decay=0.1,
        logging_steps=20,
        eval_strategy="epoch",
        # No Trainer checkpoints: the wrapper is a plain nn.Module, so Trainer would
        # save the ENTIRE state (incl. the frozen ~350M-param base, ~1.4GB) each epoch.
        # Large write bursts coincided with repeated WSL crashes on the SCX run; we
        # save only the small LoRA adapter + classifier head once, at the end.
        save_strategy="no",
        report_to=[],
        bf16=torch.cuda.is_available(),
        remove_unused_columns=False,
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset)
    print("Starting training...")
    trainer.train()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save(model.classifier.state_dict(), os.path.join(OUTPUT_DIR, "classifier_head.pt"))
    model.body.save_pretrained(os.path.join(OUTPUT_DIR, "lora_adapter"))
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved to: {OUTPUT_DIR}")

    eval_results = trainer.evaluate()
    print("Final eval:", eval_results)


if __name__ == "__main__":
    main()
