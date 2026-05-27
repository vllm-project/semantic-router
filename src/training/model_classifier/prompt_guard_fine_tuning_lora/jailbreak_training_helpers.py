"""Shared helpers for jailbreak LoRA training orchestration."""

import json
import os

import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import TrainingArguments


def split_training_data(sample_data: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split prepared samples into train and validation partitions."""
    train_size = int(0.8 * len(sample_data))
    return sample_data[:train_size], sample_data[train_size:]


def compute_security_metrics(eval_pred) -> dict[str, float]:
    """Compute binary security classification metrics."""
    predictions, labels = eval_pred
    predictions = torch.argmax(torch.tensor(predictions), dim=1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary"
    )
    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def create_security_training_args(
    output_dir: str, num_epochs: int, batch_size: int, learning_rate: float
) -> TrainingArguments:
    """Create the TrainingArguments for jailbreak LoRA fine-tuning."""
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        max_grad_norm=0.0,
        lr_scheduler_type="cosine",
        warmup_ratio=0.06,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=2,
        report_to=[],
        fp16=False,
        dataloader_drop_last=False,
        eval_accumulation_steps=1,
    )


def save_training_artifacts(
    output_dir: str,
    model,
    tokenizer,
    label_to_id: dict,
    id_to_label: dict,
    lora_config: dict,
    logger,
) -> None:
    """Persist LoRA adapters plus label/config metadata."""
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    label_mapping_data = {
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
    }
    with open(os.path.join(output_dir, "label_mapping.json"), "w") as f:
        json.dump(label_mapping_data, f)

    with open(os.path.join(output_dir, "jailbreak_type_mapping.json"), "w") as f:
        json.dump(label_mapping_data, f)
    logger.info("Created jailbreak_type_mapping.json for Go testing compatibility")

    with open(os.path.join(output_dir, "lora_config.json"), "w") as f:
        json.dump(lora_config, f)


def log_training_summary(
    eval_results: dict, output_dir: str, model_path: str, logger
) -> None:
    """Log evaluation metrics and saved model paths after training completes."""
    logger.info("Validation Results:")
    logger.info(f"  Accuracy: {eval_results['eval_accuracy']:.4f}")
    logger.info(f"  F1: {eval_results['eval_f1']:.4f}")
    logger.info(f"  Precision: {eval_results['eval_precision']:.4f}")
    logger.info(f"  Recall: {eval_results['eval_recall']:.4f}")
    logger.info(f"LoRA Security model saved to: {output_dir}")
    logger.info(f"LoRA adapter saved to: {output_dir}")
    logger.info(f"Base model: {model_path} (not merged - adapters kept separate)")
