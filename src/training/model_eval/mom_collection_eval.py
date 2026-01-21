"""
MoM Collection Evaluation Script
================================

A unified evaluation script for the Mixture of Models (MoM) collection.

Supports:
- Text Classification (Feedback, Jailbreak, Fact-check, Intent)
- Token Classification (PII Detection)
- Merged Models and LoRA Adapters
- Local Datasets and HuggingFace Hub Datasets
- Custom Model Paths (for fine-tuned local models)
- Batch Processing for High Performance

Usage:
    # Evaluate official model
    python src/training/model_eval/mom_collection_eval.py --model feedback --device cuda

    # Evaluate local fine-tuned model
    python src/training/model_eval/mom_collection_eval.py --model feedback --model_id ./my-local-model
"""

import argparse
import json
import time
import logging
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Any, Union

# Metrics
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from seqeval.metrics import accuracy_score as seqe_accuracy_score
from seqeval.metrics import f1_score as seq_f1_score
from seqeval.metrics import classification_report as seq_classification_report

# HF & Modeling
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    PreTrainedTokenizer,
    PreTrainedModel,
)
from peft import PeftModel, PeftConfig

# config logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


MODEL_REGISTRY = {
    #               Text Classification Models
    "feedback": {
        "id": "llm-semantic-router/mmbert-feedback-detector-merged",
        "lora_id": "llm-semantic-router/mmbert-feedback-detector-lora",
        "type": "text_classification",
        "hf_dataset": "llm-semantic-router/feedback-dataset",
        "labels": ["SATISFIED", "FRUSTRATED", "NEUTRAL"],
    },
    "jailbreak": {
        "id": "llm-semantic-router/mmbert-jailbreak-detector-merged",
        "lora_id": "llm-semantic-router/mmbert-jailbreak-detector-lora",
        "type": "text_classification",
        "hf_dataset": "llm-semantic-router/jailbreak-dataset",
        "labels": ["BENIGN", "JAILBREAK"],
    },
    "fact-check": {
        "id": "llm-semantic-router/mmbert-fact-check-merged",
        "lora_id": "llm-semantic-router/mmbert-fact-check-lora",
        "type": "text_classification",
        "hf_dataset": "llm-semantic-router/fact-check-dataset",
        "labels": ["NO_FACT_CHECK", "FACT_CHECK_NEEDED"],
    },
    "intent": {
        "id": "llm-semantic-router/mmbert-intent-classifier-merged",
        "lora_id": "llm-semantic-router/mmbert-intent-classifier-lora",
        "type": "text_classification",
        "hf_dataset": "llm-semantic-router/intent-dataset",
        "labels": [
            "biology",
            "business",
            "chemistry",
            "cs",
            "economics",
            "engineering",
            "health",
            "history",
            "law",
            "math",
            "philosophy",
            "physics",
            "psychology",
            "other",
        ],
    },
    #                 Token Classification Models
    "pii": {
        "id": "llm-semantic-router/mmbert-pii-detector-merged",
        "lora_id": "llm-semantic-router/mmbert-pii-detector-lora",
        "type": "token_classification",
        "hf_dataset": "llm-semantic-router/pii-dataset",
        "labels": ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"],
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MoM Collection Models")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Task type to evaluate (e.g., feedback, pii).",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="Optional: Override model path (e.g., local/path or user/repo). Defaults to official registry ID.",
    )
    parser.add_argument("--use_lora", action="store_true", help="Use the LoRA variant")
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Inference batch size"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Max samples to evaluate"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="src/training/model_eval/results",
        help="Output directory",
    )
    parser.add_argument(
        "--custom_dataset", type=str, default=None, help="Path to local dataset"
    )

    return parser.parse_args()


def load_model(key: str, override_model_id: str, use_lora: bool, device: str):
    config = MODEL_REGISTRY[key]

    # Determine which model ID to use
    if override_model_id:
        model_id = override_model_id
        logger.info(f"Using Custom Model ID: {model_id}")
    else:
        model_id = config["lora_id"] if use_lora else config["id"]
        logger.info(f"Using Registry Model ID: {model_id}")

    task_type = config["type"]
    logger.info(f"Loading model on {device} (Task: {task_type})...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception:
        logger.warning(
            f"Could not load tokenizer from {model_id}. Trying base model logic..."
        )
        peft_config = PeftConfig.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)

    if use_lora:
        peft_config = PeftConfig.from_pretrained(model_id)
        base_model_id = peft_config.base_model_name_or_path

        logger.info(f"Detected LoRA. Loading base model: {base_model_id}")

        if task_type == "text_classification":
            base_model = AutoModelForSequenceClassification.from_pretrained(
                base_model_id, num_labels=len(config["labels"])
            )
        elif task_type == "token_classification":
            base_model = AutoModelForTokenClassification.from_pretrained(
                base_model_id, num_labels=len(config["labels"])
            )

        # Load Adapter
        model = PeftModel.from_pretrained(base_model, model_id)

    else:
        if task_type == "text_classification":
            model = AutoModelForSequenceClassification.from_pretrained(model_id)
        elif task_type == "token_classification":
            model = AutoModelForTokenClassification.from_pretrained(model_id)

    model.to(device)
    model.eval()
    return model, tokenizer


def load_eval_data(key: str, custom_path: str = None, limit: int = None):
    """
    Loads data from HF hub or local file.
    Returns a dataset object.
    """
    config = MODEL_REGISTRY[key]

    # custom dataset
    if custom_path:
        logger.info(f"Loading custom dataset from: {custom_path}")
        if custom_path.endswith(".json"):
            dataset = load_dataset("json", data_files=custom_path, split="train")
        elif custom_path.endswith(".csv"):
            dataset = load_dataset("csv", data_files=custom_path, split="train")

        if limit:
            dataset = dataset.select(range(min(len(dataset), limit)))
        return dataset

    # hf dataset
    try:
        hf_id = config.get("hf_dataset")
        logger.info(f"Attempting to load real dataset: {hf_id}...")

        dataset = load_dataset(hf_id, split="test")
        logger.info(f"Successfully loaded {len(dataset)} samples from {hf_id}")

        if limit:
            dataset = dataset.select(range(min(len(dataset), limit)))

        return dataset

    except Exception as e:
        # fallback to dummy data
        logger.warning(
            f"Could not load HF dataset '{config.get('hf_dataset')}' (Error: {str(e)})."
        )
        logger.warning("Falling back to GENERATED DUMMY DATA for pipeline validation.")
        return generate_dummy_data(config["type"], config["labels"], limit or 10)


def generate_dummy_data(task_type, labels, count):
    data = []
    if task_type == "text_classification":
        for i in range(count):
            data.append(
                {
                    "text": f"This is a test sentence number {i} for evaluation.",
                    "label": i % len(labels),
                }
            )
    elif task_type == "token_classification":
        for i in range(count):
            data.append(
                {
                    "tokens": ["This", "is", "a", "test", "sentence", "."],
                    "ner_tags": [0, 0, 0, 0, 0, 0],
                }
            )

    return Dataset.from_list(data)


#      Eval Loops


def evaluate_text_classification(model, tokenizer, dataset, device, labels, batch_size):
    predictions = []
    ground_truths = []
    latencies = []

    logger.info(f"Starting Text Classification Evaluation (Batch Size: {batch_size})")

    # Batched iteration
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i : i + batch_size]

        if isinstance(batch, dict):
            texts = batch.get("text", batch.get("sentence"))
            true_labels = batch["label"]
        else:
            texts = [item.get("text", item.get("sentence")) for item in batch]
            true_labels = [item["label"] for item in batch]

        start_ts = time.time()

        inputs = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

            preds_batch = torch.argmax(logits, dim=-1).cpu().tolist()

        batch_latency = (time.time() - start_ts) * 1000
        latencies.extend([batch_latency / len(texts)] * len(texts))

        predictions.extend(preds_batch)
        ground_truths.extend(true_labels)

    return predictions, ground_truths, latencies


def evaluate_token_classification(
    model, tokenizer, dataset, device, label_map, batch_size
):
    predictions = []
    ground_truths = []
    latencies = []

    logger.info(f"Starting Token Classification Evaluation (Batch Size: {batch_size})")

    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i : i + batch_size]

        if isinstance(batch, dict):
            batch_words = batch["tokens"]
            batch_tags = batch["ner_tags"]
        else:
            batch_words = [item["tokens"] for item in batch]
            batch_tags = [item["ner_tags"] for item in batch]

        start_ts = time.time()

        # Tokenize batch
        tokenized_inputs = tokenizer(
            batch_words,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        with torch.no_grad():
            outputs = model(**tokenized_inputs)
            logits = outputs.logits
            pred_ids = torch.argmax(logits, dim=-1)

        batch_latency = (time.time() - start_ts) * 1000
        latencies.extend([batch_latency / len(batch_words)] * len(batch_words))

        # Align predictions for each sample in the batch
        pred_ids = pred_ids.cpu().tolist()

        for idx, pred_row in enumerate(pred_ids):
            word_ids = tokenized_inputs.word_ids(batch_index=idx)
            true_tag_ids = batch_tags[idx]

            aligned_preds = []
            aligned_truth = []

            previous_word_idx = None
            for token_idx, word_idx in enumerate(word_ids):
                if word_idx is None:
                    continue
                elif word_idx != previous_word_idx:
                    # Start of new word
                    if token_idx < len(pred_row):
                        aligned_preds.append(label_map[pred_row[token_idx]])
                    else:
                        aligned_preds.append("O")  # Fallback

                    if word_idx < len(true_tag_ids):
                        aligned_truth.append(label_map[true_tag_ids[word_idx]])
                    else:
                        aligned_truth.append("O")

                previous_word_idx = word_idx

            predictions.append(aligned_preds)
            ground_truths.append(aligned_truth)

    return predictions, ground_truths, latencies


def calculate_metrics(predictions, ground_truths, latencies, labels, task_type):
    metrics = {
        "latency_ms": {
            "avg": np.mean(latencies),
            "p50": np.percentile(latencies, 50),
            "p99": np.percentile(latencies, 99),
        }
    }

    if task_type == "text_classification":
        acc = accuracy_score(ground_truths, predictions)
        prec, rec, f1, _ = precision_recall_fscore_support(
            ground_truths, predictions, average="weighted", zero_division=0
        )
        metrics["accuracy"] = acc
        metrics["precision"] = prec
        metrics["recall"] = rec
        metrics["f1"] = f1

        # Confusion Matrix
        cm = confusion_matrix(ground_truths, predictions, labels=range(len(labels)))
        metrics["confusion_matrix"] = cm.tolist()

    elif task_type == "token_classification":
        acc = seqe_accuracy_score(ground_truths, predictions)
        f1 = seq_f1_score(ground_truths, predictions)
        report = seq_classification_report(ground_truths, predictions, output_dict=True)

        metrics["token_accuracy"] = acc
        metrics["f1"] = f1
        metrics["detailed_report"] = report

    return metrics


def save_report(args, metrics, labels, task_type):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    #         Save JSON
    json_path = output_dir / f"{args.model}_eval_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {json_path}")

    #       Save Confusion Matrix (Text Class only)
    if task_type == "text_classification" and "confusion_matrix" in metrics:
        cm = np.array(metrics["confusion_matrix"])
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix: {args.model}")

        plot_path = output_dir / f"{args.model}_cm_{timestamp}.png"
        plt.savefig(plot_path)
        logger.info(f"Confusion matrix plot saved to {plot_path}")

    # Print Summary
    print("\n" + "=" * 40)
    print(f"EVALUATION REPORT: {args.model}")
    print("=" * 40)
    print(f"Accuracy: {metrics.get('accuracy', metrics.get('token_accuracy', 0)):.4f}")
    if "f1" in metrics:
        print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Avg Latency: {metrics['latency_ms']['avg']:.2f} ms")
    print("=" * 40 + "\n")


#                                 --- MAIN ---


def main():
    args = parse_args()
    config = MODEL_REGISTRY[args.model]

    model, tokenizer = load_model(args.model, args.model_id, args.use_lora, args.device)
    dataset = load_eval_data(args.model, args.custom_dataset, args.limit)

    if config["type"] == "text_classification":
        preds, truths, lats = evaluate_text_classification(
            model, tokenizer, dataset, args.device, config["labels"], args.batch_size
        )
    else:
        preds, truths, lats = evaluate_token_classification(
            model, tokenizer, dataset, args.device, config["labels"], args.batch_size
        )

    metrics = calculate_metrics(preds, truths, lats, config["labels"], config["type"])
    save_report(args, metrics, config["labels"], config["type"])


if __name__ == "__main__":
    main()
