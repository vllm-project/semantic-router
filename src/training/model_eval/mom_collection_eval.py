"""
MoM Collection Evaluation Script
================================

A unified evaluation script for the Mixture of Models (MoM) collection.

Supports:
- Text Classification (Feedback, Jailbreak, Fact-check, Intent)
- Token Classification (PII Detection)
- Merged Models and LoRA Adapters
- Local Datasets and HuggingFace Hub Datasets

Usage:
    python src/training/model_eval/mom_collection_eval.py --model feedback --device cuda
    python src/training/model_eval/mom_collection_eval.py --model fact-check --use_lora
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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
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
    PreTrainedModel
)
from peft import PeftModel, PeftConfig

# config logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


MODEL_REGISTRY = {
    #                          Text Classification Models
    "feedback": {
        "id": "llm-semantic-router/mmbert-feedback-detector-merged",
        "lora_id": "llm-semantic-router/mmbert-feedback-detector-lora",
        "type": "text_classification",
        "hf_dataset": "llm-semantic-router/feedback-dataset",
        "labels": ["SATISFIED", "FRUSTRATED", "NEUTRAL"]
    },
    "jailbreak": {
        "id": "llm-semantic-router/mmbert-jailbreak-detector-merged",
        "lora_id": "llm-semantic-router/mmbert-jailbreak-detector-lora",
        "type": "text_classification",
        "hf_dataset": "llm-semantic-router/jailbreak-dataset",
        "labels": ["BENIGN", "JAILBREAK"]
    },
    "fact-check": {
        "id": "llm-semantic-router/mmbert-fact-check-merged",
        "lora_id": "llm-semantic-router/mmbert-fact-check-lora",
        "type": "text_classification",
        "hf_dataset": "llm-semantic-router/fact-check-dataset",
        "labels": ["NO_FACT_CHECK", "FACT_CHECK_NEEDED"]
    },
    "intent": {
        "id": "llm-semantic-router/mmbert-intent-classifier-merged",
        "lora_id": "llm-semantic-router/mmbert-intent-classifier-lora",
        "type": "text_classification",
        "hf_dataset": "llm-semantic-router/intent-dataset",
        "labels": ["biology", "business", "chemistry", "cs", "economics",
                   "engineering", "health", "history", "law", "math", "philosophy",
                   "physics", "psychology", "other"]
    },

    #                             Token Classification Models
    "pii": {
        "id": "llm-semantic-router/mmbert-pii-detector-merged",
        "lora_id": "llm-semantic-router/mmbert-pii-detector-lora",
        "type": "token_classification",
        "hf_dataset": "llm-semantic-router/pii-dataset",
        "labels": ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
    }
}

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MoM Collection Models")
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_REGISTRY.keys()),
                        help="Model key from registry.")
    parser.add_argument("--use_lora", action="store_true", help="Use the LoRA variant")
    parser.add_argument("--batch_size", type=int, default=16, help="Inference batch size")
    parser.add_argument("--limit", type=int, default=None, help="Max samples to evaluate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="src/training/model_eval/results", help="Output directory")
    parser.add_argument("--custom_dataset", type=str, default=None, help="Path to local dataset")

    return parser.parse_args()

def load_model(key: str, use_lora: bool, device: str):
    config = MODEL_REGISTRY[key]
    model_id = config["lora_id"] if use_lora else config["id"]
    task_type = config["type"]

    logger.info(f"Loading model: {model_id} (Task: {task_type}) on {device}")
# load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception:
        logger.warning(f"Could not load tokenizer from {model_id}. Trying base model logic...")
        peft_config = PeftConfig.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
 # load model
    if use_lora:
        peft_config = PeftConfig.from_pretrained(model_id)
        base_model_id = peft_config.base_model_name_or_path

        logger.info(f"Detected LoRA. Loading base model: {base_model_id}")

        if task_type == "text_classification":
            base_model = AutoModelForSequenceClassification.from_pretrained(base_model_id, num_labels=len(config["labels"]))
        elif task_type == "token_classification":
            base_model = AutoModelForTokenClassification.from_pretrained(base_model_id, num_labels=len(config["labels"]))

        # load adapter
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

    if custom_path:
        logger.info(f"Loading custom dataset from: {custom_path}")
        if custom_path.endswith('.json'):
            dataset = load_dataset('json', data_files=custom_path, split='train')
        elif custom_path.endswith('.csv'):
            dataset = load_dataset('csv', data_files=custom_path, split='train')
        if limit:
            dataset = dataset.select(range(min(len(dataset), limit)))
        return dataset


    try:
        hf_id = config.get("hf_dataset")
        logger.info(f"Attempting to load real dataset: {hf_id}...")

        dataset = load_dataset(hf_id, split="test")
        logger.info(f"Successfully loaded {len(dataset)} samples from {hf_id}")

        if limit:
            dataset = dataset.select(range(min(len(dataset), limit)))
        return dataset

    except Exception as e:
        # dummy data
        logger.warning(
            f"Could not load HF dataset '{config.get('hf_dataset')}' (Error: {str(e)}).")
        logger.warning(
            "Falling back to GENERATED DUMMY DATA for pipeline validation.")
        return generate_dummy_data(config["type"], config["labels"], limit or 10)

def generate_dummy_data(task_type, labels, count):
    data = []
    if task_type == "text_classification":
        for i in range(count):
            data.append({
                "text": f"This is a test sentence number {i} for evaluation.",
                "label": i % len(labels)
            })
    elif task_type == "token_classification":
        for i in range(count):
            data.append({
                "tokens": ["This", "is", "a", "test", "sentence", "."],
                "ner_tags": [0, 0, 0, 0, 0, 0]
            })


    return Dataset.from_list(data)


         # Eval loops

def evaluate_text_classification(model, tokenizer, dataset, device, labels):
    predictions = []
    ground_truths = []
    latencies = []

    logger.info("Starting Text Classification Evaluation")

    for item in tqdm(dataset):
        text = item.get("text", item.get("sentence"))
        true_label = item["label"]

        start_ts = time.time()

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            pred_idx = torch.argmax(logits, dim=-1).item()

        latencies.append((time.time() - start_ts) * 1000)

        predictions.append(pred_idx)
        ground_truths.append(true_label)

    return predictions, ground_truths, latencies

def evaluate_token_classification(model, tokenizer, dataset, device, label_map):
    predictions = []
    ground_truths = []
    latencies = []

    logger.info("Starting Token Classification Evaluation")

    for item in tqdm(dataset):
        words = item["tokens"]
        true_tag_ids = item["ner_tags"]

        start_ts = time.time()

        # Tokenizing & preserving word alignment
        tokenized_inputs = tokenizer(words, is_split_into_words=True, return_tensors="pt").to(device)
        word_ids = tokenized_inputs.word_ids()

        with torch.no_grad():
            outputs = model(**tokenized_inputs)
            logits = outputs.logits
            pred_ids = torch.argmax(logits, dim=-1).squeeze().tolist()

        latencies.append((time.time() - start_ts) * 1000)

        aligned_preds = []
        aligned_truth = []

        # Align predictions back to words
        previous_word_idx = None
        for idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            elif word_idx != previous_word_idx:
                # Start of new word
                if idx < len(pred_ids):
                    aligned_preds.append(label_map[pred_ids[idx]])
                else:
                    aligned_preds.append("O")

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
            "p99": np.percentile(latencies, 99)
        }
    }

    if task_type == "text_classification":
        acc = accuracy_score(ground_truths, predictions)
        prec, rec, f1, _ = precision_recall_fscore_support(ground_truths, predictions,
                                                           average='weighted', zero_division=0)
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

    # save json
    json_path = output_dir / f"{args.model}_eval_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {json_path}")

    #  save conf mat (text class only)
    if task_type == "text_classification" and "confusion_matrix" in metrics:
        cm = np.array(metrics["confusion_matrix"])
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix: {args.model}')

        plot_path = output_dir / f"{args.model}_cm_{timestamp}.png"
        plt.savefig(plot_path)
        logger.info(f"Confusion matrix plot saved to {plot_path}")

    # print summary
    print("\n" + "="*40)
    print(f"EVALUATION REPORT: {args.model}")
    print("="*40)
    print(f"Accuracy: {metrics.get('accuracy', metrics.get('token_accuracy', 0)):.4f}")
    if 'f1' in metrics: print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Avg Latency: {metrics['latency_ms']['avg']:.2f} ms")
    print("="*40 + "\n")


#                       main

def main():
    args = parse_args()
    config = MODEL_REGISTRY[args.model]

    model, tokenizer = load_model(args.model, args.use_lora, args.device)
    dataset = load_eval_data(args.model, args.custom_dataset, args.limit)


    if config["type"] == "text_classification":
        preds, truths, lats = evaluate_text_classification(model, tokenizer, dataset, args.device, config["labels"])
    else:
        preds, truths, lats = evaluate_token_classification(model, tokenizer, dataset, args.device, config["labels"])

    metrics = calculate_metrics(preds, truths, lats, config["labels"], config["type"])
    save_report(args, metrics, config["labels"], config["type"])

if __name__ == "__main__":
    main()
