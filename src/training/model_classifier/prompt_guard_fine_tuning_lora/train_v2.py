"""Train the injection-specific mmBERT-32K candidate from prepared v2 data.

This entry point leaves the old toxicity-mixed recipe available for historical
reproduction. Only validation selects checkpoints; external final tests are
scored separately after the candidate is frozen.
"""

from __future__ import annotations

# Training dependencies remain optional for dataset contract checks.
# ruff: noqa: PLC0415
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

BASE_ID = "llm-semantic-router/mmbert-32k-yarn"
BASE_REVISION = "72a23a6640489471eb4ff7ad3ec5bc80af8a27de"
LABEL2ID = {"benign": 0, "jailbreak": 1}
CONTRACT_VERSION = 2


def read_prepared(directory: Path) -> tuple[dict, dict]:
    manifest = json.loads((directory / "data_manifest.json").read_text())
    if (
        manifest.get("contract_version") != CONTRACT_VERSION
        or manifest.get("label2id") != LABEL2ID
    ):
        raise ValueError("Expected injection-specific v2 data and labels")
    data = {}
    for split in ("train", "validation"):
        path = directory / f"{split}.jsonl"
        if (
            hashlib.sha256(path.read_bytes()).hexdigest()
            != manifest["splits"][split]["sha256"]
        ):
            raise ValueError(f"Prepared {split} content no longer matches manifest")
        with path.open(encoding="utf-8") as handle:
            data[split] = [json.loads(line) for line in handle if line.strip()]
    return data, manifest


def train(args) -> None:
    import numpy as np
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from transformers import (
        AutoConfig,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    records, data_manifest = read_prepared(args.data_dir)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Use an empty output directory for each candidate")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(42)
    config = AutoConfig.from_pretrained(BASE_ID, revision=BASE_REVISION)
    if not 0 < args.max_length <= config.max_position_embeddings:
        raise ValueError("Training budget exceeds the actual backbone context")
    config.reference_compile = False
    config.num_labels = 2
    config.id2label = {value: key for key, value in LABEL2ID.items()}
    config.label2id = LABEL2ID
    tokenizer = AutoTokenizer.from_pretrained(BASE_ID, revision=BASE_REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_ID,
        revision=BASE_REVISION,
        config=config,
        torch_dtype=torch.float32,
        attn_implementation="sdpa",
    )
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=32,
            lora_alpha=64,
            lora_dropout=0.1,
            target_modules=["attn.Wqkv", "attn.Wo", "mlp.Wi", "mlp.Wo"],
            modules_to_save=["classifier"],
            bias="none",
            revision=BASE_REVISION,
        ),
    )

    def tokenize(batch):
        encoded = tokenizer(
            batch["text"], truncation=True, max_length=args.max_length, padding=False
        )
        encoded["labels"] = batch["label"]
        return encoded

    datasets = {}
    for split, rows in records.items():
        dataset = Dataset.from_list(rows)
        datasets[split] = dataset.map(
            tokenize, batched=True, remove_columns=dataset.column_names
        )

    def metrics(prediction):
        logits, labels = prediction
        predicted = np.argmax(logits, axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predicted, average="binary", zero_division=0
        )
        _, _, macro, _ = precision_recall_fscore_support(
            labels, predicted, average="macro", zero_division=0
        )
        return {
            "accuracy": accuracy_score(labels, predicted),
            "f1_macro": macro,
            "precision_injection": precision,
            "recall_injection": recall,
            "f1_injection": f1,
        }

    training = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_grad_norm=1.0,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=10,
        bf16=torch.cuda.is_available(),
        report_to=[],
        seed=42,
        data_seed=42,
        dataloader_num_workers=4,
    )
    trainer = Trainer(
        model=model,
        args=training,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8),
        compute_metrics=metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    # ModernBERT returns mean CE; it does not use num_items_in_batch.
    trainer.model_accepts_loss_kwargs = False
    result = trainer.train()
    validation = trainer.evaluate()
    adapter = args.output_dir / "adapter"
    trainer.save_model(str(adapter))
    tokenizer.save_pretrained(adapter)
    merged = args.output_dir / "merged"
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(merged)
    tokenizer.save_pretrained(merged)
    recipe = {
        "contract_version": 2,
        "task": "prompt-injection-and-jailbreak",
        "base_model": BASE_ID,
        "base_revision": BASE_REVISION,
        "max_length": args.max_length,
        "backbone_capacity": config.max_position_embeddings,
        "label2id": LABEL2ID,
        "id2label": config.id2label,
        "epochs_limit": args.epochs,
        "seed": 42,
        "global_batch": 64,
        "learning_rate": 1e-4,
        "lora_rank": 32,
        "lora_alpha": 64,
        "selection": "validation macro F1",
        "final_test_evaluated": False,
        "semantics": "benign means no instruction attack; harmful content requires the separate safety classifier",
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "peft", "datasets")
        },
        "train_metrics": result.metrics,
        "validation_metrics": validation,
    }
    for directory in (adapter, merged):
        for name, value in (
            ("training_config.json", recipe),
            ("data_manifest.json", data_manifest),
            ("label_mapping.json", {"label2id": LABEL2ID, "id2label": config.id2label}),
            (
                "jailbreak_type_mapping.json",
                {"label_to_id": LABEL2ID, "id_to_label": config.id2label},
            ),
        ):
            (directory / name).write_text(
                json.dumps(value, indent=2) + "\n", encoding="utf-8"
            )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(recipe, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=5)
    train(parser.parse_args())
