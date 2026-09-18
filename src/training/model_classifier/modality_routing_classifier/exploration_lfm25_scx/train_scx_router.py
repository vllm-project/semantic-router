"""Curiosity exploration: fine-tune SCX Router v0.1 (GLiClass, decoder-KV Qwen3-0.6B
backbone) on the modality-routing task, using gliclass's own native training pipeline
(full fine-tune, not LoRA -- PEFT compatibility with this custom nested architecture
wasn't verified, and this is exploratory, not decision-record-quality work)."""
import json
import os
import sys

import torch
from gliclass import GLiClassModel
from gliclass.data_processing import AugmentationConfig, DataCollatorWithPadding, GLiClassDataset
from gliclass.training import Trainer, TrainingArguments
from transformers import AutoTokenizer
from pathlib import Path as _Path

_HERE = _Path(__file__).resolve().parent
_CLASSIFIER_DIR = _HERE.parent  # modality_routing_classifier/
_DATA_DIR = _CLASSIFIER_DIR / "exported_modality_routing_dataset"


MODEL_ID = "scx-admin/scx-router-v0.1"
DATA_DIR = str(_DATA_DIR)
OUTPUT_DIR = os.environ.get("SCX_OUTPUT_DIR", str(_HERE / "runs" / "scx_router_finetuned"))

LABELS = ["AR", "DIFFUSION", "BOTH"]
LABEL_DESCRIPTIONS = {
    "AR": "a text-only response",
    "DIFFUSION": "generating an image",
    "BOTH": "both a text explanation and an image",
}


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def to_gliclass_examples(rows):
    return [
        {
            "text": row["text"],
            "all_labels": LABELS,
            "true_labels": [row["label_name"]],
        }
        for row in rows
    ]


# NOTE: this checkpoint's decoder-KV architecture only implements a working
# get_loss() path for its native problem_type ("multi_label_classification" --
# per-label independent BCE/focal loss, where labels are represented as a
# multi-hot vector aligned to the <<LABEL>> token positions). Its
# "single_label_classification" loss path is broken for this architecture as
# shipped (labels[:, :num_labels] assumes a 2D tensor that a scalar class index
# never produces, and the collator separately drops 0-dim label tensors
# entirely) -- verified by direct reproduction, not assumed. Since our task is
# mutually-exclusive 3-class, we frame it as multi-hot with exactly one active
# label per example, which is what the model natively (and correctly) supports.
print("Loading model + tokenizer...", file=sys.stderr)
model = GLiClassModel.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
print(f"Native problem_type: {model.config.problem_type}", file=sys.stderr)

train_rows = load_jsonl(f"{DATA_DIR}/train.jsonl")
val_rows = load_jsonl(f"{DATA_DIR}/validation.jsonl")
print(f"Train: {len(train_rows)}, Val: {len(val_rows)}", file=sys.stderr)

augment_config = AugmentationConfig(enabled=False)
label2description = {name: f"{name}: {desc}" for name, desc in LABEL_DESCRIPTIONS.items()}

train_dataset = GLiClassDataset(
    to_gliclass_examples(train_rows),
    tokenizer,
    augment_config,
    label2description=label2description,
    max_length=512,
    problem_type="multi_label_classification",
    architecture_type="decoder-kv",
)
val_dataset = GLiClassDataset(
    to_gliclass_examples(val_rows),
    tokenizer,
    augment_config,
    label2description=label2description,
    max_length=512,
    problem_type="multi_label_classification",
    architecture_type="decoder-kv",
)

data_collator = DataCollatorWithPadding(device="cuda:0", config=model.config)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.1,
    logging_steps=20,
    eval_strategy="epoch",
    # Two consecutive WSL crashes both hit right after the first epoch's 3.4GB
    # checkpoint write (2.5GB weights + 1.2GB optimizer state) -- no intermediate
    # checkpoints, and no optimizer state saved, to avoid large write bursts.
    save_strategy="no",
    save_optimizer_state=False,
    report_to="none",
    bf16=torch.cuda.is_available(),
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

print("Starting training...", file=sys.stderr)
trainer.train()

model.to(torch.bfloat16).save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Saved fine-tuned SCX Router to: {OUTPUT_DIR}")
