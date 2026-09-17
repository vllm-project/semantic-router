"""
Three-Way Evaluation for the Modality Routing Candidate Experiment (Issue #3198)
=================================================================================

Evaluates the published production baseline, a leakage-free "clean baseline"
reproduction, and the distilled candidate against export_modality_dataset.py's
deterministic test.jsonl, and reports:

  1. A test-set contamination check: whether a test row's text (exact or
     whitespace/case-normalized) also appears in train.jsonl/validation.jsonl.
     Motivated by a measured finding, not speculation -- one of the real DIFFUSION-
     class source datasets (Gustavosta/Stable-Diffusion-Prompts) was found to have
     a 32.44% exact-duplicate row rate, and the exporter's stratified split does not
     deduplicate before assigning train/val/test membership. See DECISION_RECORD.md.
  2. Standard classification metrics (accuracy, weighted F1, per-class P/R/F1,
     confusion matrix) for each model, computed on both the full test set and the
     contamination-filtered subset.
  3. Three pairwise routing-agreement metrics (does the pair predict the same label;
     on disagreement, which side matched ground truth), also full and filtered:
       - candidate vs clean_baseline   (primary, leakage-free graduation-gate pair)
       - candidate vs published_baseline (secondary/reference; what #3856's harness
         will eventually re-run with real latency numbers attached)
       - clean_baseline vs published_baseline (quantifies how much the cross-run
         leakage risk actually mattered, empirically)

Same-run p99 latency (#3856) and fixed-policy control comparison (#3857) are
explicitly out of scope here -- see the `pending_dependencies` block in the report.

Usage:
    python evaluate_modality_candidate.py \\
        --test-file exported_modality_routing_dataset/test.jsonl \\
        --train-file exported_modality_routing_dataset/train.jsonl \\
        --val-file exported_modality_routing_dataset/validation.jsonl \\
        --published-baseline-model-path llm-semantic-router/mmbert32k-modality-router-merged \\
        --clean-baseline-model-path models/mmbert32k-modality-router-clean-merged \\
        --candidate-model-path models/distilbert-modality-router-candidate-merged \\
        --output-report modality_candidate_eval_report.json
"""

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_THIS_DIR)
sys.path.append(os.path.dirname(_THIS_DIR))

from common_lora_utils import load_sequence_classifier_for_inference, setup_logging
from modality_routing_bert_finetuning_lora import MODALITY_LABELS

logger = setup_logging()

MODEL_KEYS = ["published_baseline", "clean_baseline", "candidate"]
AGREEMENT_PAIRS = [
    ("candidate", "clean_baseline"),
    ("candidate", "published_baseline"),
    ("clean_baseline", "published_baseline"),
]


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def sha256_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def flag_test_contamination(
    train_texts: List[str], val_texts: List[str], test_rows: List[Dict]
) -> Tuple[List[int], Dict[str, float]]:
    """
    Returns (contaminated_row_indices, contamination_rate_by_class) where a test row
    is "contaminated" if its exact text or normalized text also appears in
    train.jsonl/validation.jsonl.
    """
    train_val_exact = set(train_texts) | set(val_texts)
    train_val_normalized = {normalize_text(t) for t in train_texts + val_texts}

    contaminated_indices = []
    counts_by_class: Dict[str, int] = {}
    contaminated_by_class: Dict[str, int] = {}

    for idx, row in enumerate(test_rows):
        label_name = row["label_name"]
        counts_by_class[label_name] = counts_by_class.get(label_name, 0) + 1
        text = row["text"]
        is_contaminated = text in train_val_exact or normalize_text(text) in train_val_normalized
        if is_contaminated:
            contaminated_indices.append(idx)
            contaminated_by_class[label_name] = contaminated_by_class.get(label_name, 0) + 1

    rate_by_class = {
        label_name: round(contaminated_by_class.get(label_name, 0) / count, 4)
        for label_name, count in counts_by_class.items()
    }
    return contaminated_indices, rate_by_class


def run_inference(model_path: str, texts: List[str], batch_size: int, max_length: int) -> np.ndarray:
    """Returns predicted label ids [N], in the order of `texts`."""
    model, tokenizer, id2label = load_sequence_classifier_for_inference(
        model_path, num_labels=len(MODALITY_LABELS)
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # id2label from the checkpoint may not use the canonical MODALITY_LABELS order
    # (AR=0/DIFFUSION=1/BOTH=2) -- remap defensively rather than assume it matches.
    label_name_to_canonical_id = {label: idx for idx, label in enumerate(MODALITY_LABELS)}
    checkpoint_id_to_canonical_id = {
        ckpt_id: label_name_to_canonical_id[label_name]
        for ckpt_id, label_name in id2label.items()
        if label_name in label_name_to_canonical_id
    }

    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch, truncation=True, padding=True, max_length=max_length, return_tensors="pt"
            ).to(device)
            logits = model(**enc).logits
            batch_preds = torch.argmax(logits, dim=-1).cpu().tolist()
            preds.extend(checkpoint_id_to_canonical_id.get(p, p) for p in batch_preds)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.array(preds, dtype=np.int64)


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    labels = list(range(len(MODALITY_LABELS)))
    accuracy = float(accuracy_score(y_true, y_pred))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0))
    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=labels, zero_division=0
    )
    per_class = {
        MODALITY_LABELS[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1_per_class[i]),
            "support": int(support[i]),
        }
        for i in range(len(MODALITY_LABELS))
    }
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return {
        "accuracy": accuracy,
        "f1_weighted": f1_weighted,
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def compute_routing_agreement(
    y_true: np.ndarray, preds_a: np.ndarray, preds_b: np.ndarray
) -> Dict:
    """Per-example agreement between two models' predictions, plus who matched ground
    truth on disagreements. Order of a/b only affects labeling, not the counts."""
    n = len(y_true)
    agree_mask = preds_a == preds_b
    num_agree = int(agree_mask.sum())
    num_disagree = n - num_agree

    a_matched_truth_b_wrong = 0
    b_matched_truth_a_wrong = 0
    both_wrong_different_labels = 0
    for i in range(n):
        if agree_mask[i]:
            continue
        a_correct = preds_a[i] == y_true[i]
        b_correct = preds_b[i] == y_true[i]
        if a_correct and not b_correct:
            a_matched_truth_b_wrong += 1
        elif b_correct and not a_correct:
            b_matched_truth_a_wrong += 1
        else:
            both_wrong_different_labels += 1

    def rate_of_disagreements(count: int) -> Optional[float]:
        return round(count / num_disagree, 4) if num_disagree else None

    agreement_by_true_label = {}
    for i, label_name in enumerate(MODALITY_LABELS):
        mask = y_true == i
        if mask.sum() > 0:
            agreement_by_true_label[label_name] = float(agree_mask[mask].mean())
        else:
            agreement_by_true_label[label_name] = None

    return {
        "agreement_rate": float(num_agree / n) if n else None,
        "num_agree": num_agree,
        "num_disagree": num_disagree,
        "disagreement_breakdown": {
            "a_matched_truth_b_wrong": {
                "count": a_matched_truth_b_wrong,
                "rate_of_disagreements": rate_of_disagreements(a_matched_truth_b_wrong),
            },
            "b_matched_truth_a_wrong": {
                "count": b_matched_truth_a_wrong,
                "rate_of_disagreements": rate_of_disagreements(b_matched_truth_a_wrong),
            },
            "both_wrong_different_labels": {
                "count": both_wrong_different_labels,
                "rate_of_disagreements": rate_of_disagreements(both_wrong_different_labels),
            },
        },
        "agreement_rate_by_ground_truth_label": agreement_by_true_label,
    }


def evaluate_subset(
    y_true: np.ndarray, preds: Dict[str, np.ndarray]
) -> Dict:
    metrics = {key: compute_classification_metrics(y_true, preds[key]) for key in MODEL_KEYS}
    agreements = {}
    for a, b in AGREEMENT_PAIRS:
        agreements[f"{a}_vs_{b}"] = compute_routing_agreement(y_true, preds[a], preds[b])
    return {"metrics": metrics, "routing_agreement": agreements}


def main(
    test_file: str,
    train_file: str,
    val_file: str,
    published_baseline_model_path: str,
    clean_baseline_model_path: str,
    candidate_model_path: str,
    output_report: str,
    batch_size: int = 32,
    max_length: int = 256,
    include_full_text: bool = False,
):
    test_rows = load_jsonl(test_file)
    train_rows = load_jsonl(train_file)
    val_rows = load_jsonl(val_file)
    logger.info(f"Loaded {len(test_rows)} test rows, {len(train_rows)} train, {len(val_rows)} val")

    contaminated_indices, contamination_rate_by_class = flag_test_contamination(
        [r["text"] for r in train_rows], [r["text"] for r in val_rows], test_rows
    )
    logger.info(
        f"Contamination check: {len(contaminated_indices)}/{len(test_rows)} test rows "
        f"({len(contaminated_indices) / len(test_rows) * 100:.2f}%) have an exact/normalized "
        f"duplicate in train+val. By class: {contamination_rate_by_class}"
    )

    texts = [r["text"] for r in test_rows]
    y_true = np.array([r["label"] for r in test_rows], dtype=np.int64)

    model_paths = {
        "published_baseline": published_baseline_model_path,
        "clean_baseline": clean_baseline_model_path,
        "candidate": candidate_model_path,
    }
    preds: Dict[str, np.ndarray] = {}
    for key, path in model_paths.items():
        logger.info(f"Running inference for {key}: {path}")
        preds[key] = run_inference(path, texts, batch_size, max_length)

    full_report = evaluate_subset(y_true, preds)

    contaminated_set = set(contaminated_indices)
    clean_idx = [i for i in range(len(test_rows)) if i not in contaminated_set]
    if clean_idx:
        clean_y_true = y_true[clean_idx]
        clean_preds = {key: preds[key][clean_idx] for key in MODEL_KEYS}
        filtered_report = evaluate_subset(clean_y_true, clean_preds)
    else:
        filtered_report = None
        logger.warning("Contamination filter removed all test rows; filtered report skipped.")

    per_example_records = []
    for idx, row in enumerate(test_rows):
        record = {
            "row_index": idx,
            "input_hash_sha256": sha256_of(row["text"]),
            "ground_truth": row["label_name"],
            "contaminated": idx in contaminated_set,
        }
        for key in MODEL_KEYS:
            record[f"{key}_pred"] = MODALITY_LABELS[int(preds[key][idx])]
        if include_full_text:
            record["text"] = row["text"]
        per_example_records.append(record)

    report = {
        "metadata": {
            "test_file": test_file,
            "train_file": train_file,
            "val_file": val_file,
            "num_test_examples": len(test_rows),
            "model_paths": model_paths,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "label_order": MODALITY_LABELS,
        },
        "contamination_check": {
            "num_contaminated": len(contaminated_indices),
            "contamination_rate": round(len(contaminated_indices) / len(test_rows), 4)
            if test_rows
            else None,
            "contamination_rate_by_ground_truth_label": contamination_rate_by_class,
            "note": "A test row is flagged if its exact or whitespace/case-normalized "
            "text also appears in train.jsonl/validation.jsonl. See DECISION_RECORD.md "
            "for the source-dataset duplicate-rate EDA that motivated this check.",
        },
        "full_test_set": full_report,
        "contamination_filtered_test_set": filtered_report,
        "per_example_records": per_example_records,
        "pending_dependencies": {
            "same_run_p99_latency": "PENDING -- blocked on #3856 (same-run evaluation "
            "harness); not computed by this script.",
            "fixed_policy_control_comparison": "PENDING -- blocked on #3857 (fixed-policy "
            "controls); not computed by this script.",
        },
    }

    with open(output_report, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report written to: {output_report}")

    logger.info("=" * 70)
    logger.info("Summary (full test set):")
    for key in MODEL_KEYS:
        logger.info(f"  {key}: accuracy={full_report['metrics'][key]['accuracy']:.4f}")
    for a, b in AGREEMENT_PAIRS:
        rate = full_report["routing_agreement"][f"{a}_vs_{b}"]["agreement_rate"]
        logger.info(f"  agreement {a} vs {b}: {rate:.4f}")
    logger.info("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Three-way modality routing candidate evaluation")
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--val-file", required=True)
    parser.add_argument("--published-baseline-model-path", required=True)
    parser.add_argument("--clean-baseline-model-path", required=True)
    parser.add_argument("--candidate-model-path", required=True)
    parser.add_argument("--output-report", default="modality_candidate_eval_report.json")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--include-full-text", action="store_true")
    args = parser.parse_args()

    main(
        test_file=args.test_file,
        train_file=args.train_file,
        val_file=args.val_file,
        published_baseline_model_path=args.published_baseline_model_path,
        clean_baseline_model_path=args.clean_baseline_model_path,
        candidate_model_path=args.candidate_model_path,
        output_report=args.output_report,
        batch_size=args.batch_size,
        max_length=args.max_length,
        include_full_text=args.include_full_text,
    )
