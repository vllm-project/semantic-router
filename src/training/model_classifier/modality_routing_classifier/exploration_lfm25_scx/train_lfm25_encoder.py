"""Curiosity exploration: fine-tune LiquidAI/LFM2.5-Encoder-350M (custom
Lfm2BidirectionalModel, interleaved gated-conv + grouped-query-attention blocks)
on the modality-routing task via LoRA.

Key gotcha (verified by direct reproduction, matching the trap @adaamko flagged in
DECISION_RECORD.md): the model card's documented
`AutoModel.from_pretrained(repo, trust_remote_code=True)` path loads a COMPLETELY
RANDOMLY-INITIALIZED model for this checkpoint. Its real weights are stored under an
`lfm2.` prefix that `Lfm2BidirectionalModel`'s own module structure doesn't expect.
The fix is to load via `AutoModelForMaskedLM.from_pretrained(...).lfm2`, which loads
the pretrained weights correctly (verified: zero MISSING/UNEXPECTED keys). See
lfm25_classifier.load_lfm25_body.

No AutoModelForSequenceClassification variant exists for this architecture, so this
wraps the bare encoder body with a masked-mean-pool + linear head, mirroring the
FocalLoss/class-weight recipe used elsewhere in this pipeline for consistency.

Usage:
    python train_lfm25_encoder.py --output-dir runs/lfm25_encoder_finetuned
"""

import argparse
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import Trainer, TrainingArguments, set_seed

from exploration_common import (
    DATA_DIR,
    LFM25_MODEL_ID,
    MODALITY_LABELS,
    RUNS_DIR,
    load_jsonl,
)
from lfm25_classifier import (
    LORA_TARGET_MODULES,
    Lfm2ForModalityClassification,
    load_lfm25_body,
    load_lfm25_tokenizer,
)
from modality_data import compute_class_stats

LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1


def add_lora(body):
    """Put a LoRA adapter on the encoder body.

    Args:
        body: The pretrained encoder.

    Returns:
        The body wrapped by PEFT, with only the adapter trainable.
    """
    config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    return get_peft_model(body, config)


def class_weights_tensor(rows: list[dict]) -> torch.Tensor:
    """Compute the class weights of a training set, using the shared rule.

    Args:
        rows: Training rows with an integer "label".

    Returns:
        A float32 tensor with one weight per class.
    """
    stats = compute_class_stats(
        [row["label"] for row in rows], num_classes=len(MODALITY_LABELS)
    )
    return torch.tensor(stats.class_weights, dtype=torch.float32)


def tokenize_rows(rows: list[dict], tokenizer, max_length: int = 256) -> Dataset:
    """Tokenize rows into a padded dataset with input ids, mask and labels.

    Args:
        rows: Rows with "text" and an integer "label".
        tokenizer: The encoder's tokenizer.
        max_length: Token limit; longer texts are truncated.

    Returns:
        A Hugging Face Dataset.
    """
    enc = tokenizer(
        [r["text"] for r in rows],
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return Dataset.from_dict(
        {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": [r["label"] for r in rows],
        }
    )


def build_training_args(args: argparse.Namespace) -> TrainingArguments:
    """Build the TrainingArguments for the run.

    Args:
        args: Parsed command-line arguments.

    Returns:
        The TrainingArguments.
    """
    return TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
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
        seed=args.seed,
        data_seed=args.seed,
    )


def save_outputs(
    model: Lfm2ForModalityClassification, tokenizer, output_dir: str
) -> None:
    """Save the classifier head, the LoRA adapter and the tokenizer.

    Args:
        model: The trained wrapper.
        tokenizer: The encoder's tokenizer.
        output_dir: Directory to write into.
    """
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.classifier.state_dict(), Path(output_dir) / "classifier_head.pt")
    model.body.save_pretrained(Path(output_dir) / "lora_adapter")
    tokenizer.save_pretrained(output_dir)


def main(args: argparse.Namespace) -> None:
    """Fine-tune the encoder with LoRA and save the adapter and head.

    Args:
        args: Parsed command-line arguments.
    """
    # Seed before the LoRA adapter and the head are initialised.
    set_seed(args.seed)
    print(
        f"Loading {LFM25_MODEL_ID} via AutoModelForMaskedLM (not the documented AutoModel path)..."
    )
    tokenizer = load_lfm25_tokenizer()
    body = add_lora(load_lfm25_body())
    body.print_trainable_parameters()

    train_rows = load_jsonl(args.train_file)
    val_rows = load_jsonl(args.val_file)
    print(f"Train: {len(train_rows)}, Val: {len(val_rows)}")
    class_weights = class_weights_tensor(train_rows)
    print(f"Class weights: {class_weights.tolist()}")

    model = Lfm2ForModalityClassification(
        body, num_labels=len(MODALITY_LABELS), class_weights=class_weights
    )
    trainer = Trainer(
        model=model,
        args=build_training_args(args),
        train_dataset=tokenize_rows(train_rows, tokenizer),
        eval_dataset=tokenize_rows(val_rows, tokenizer),
    )
    print("Starting training...")
    trainer.train()
    save_outputs(model, tokenizer, args.output_dir)
    print(f"Saved to: {args.output_dir}")
    print("Final eval:", trainer.evaluate())


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--train-file", default=str(DATA_DIR / "train.jsonl"))
    parser.add_argument("--val-file", default=str(DATA_DIR / "validation.jsonl"))
    parser.add_argument(
        "--output-dir", default=str(RUNS_DIR / "lfm25_encoder_finetuned")
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
