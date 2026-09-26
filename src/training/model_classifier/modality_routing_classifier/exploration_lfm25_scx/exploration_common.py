"""Shared pieces of the LFM2.5 and SCX exploration scripts.

Labels, model revisions, the GLiClass dataset builder and the way results are
summarised and saved live here once, so the training, evaluation and zero-shot
scripts do not each carry their own copy.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

EXPLORATION_DIR = Path(__file__).resolve().parent
CLASSIFIER_DIR = EXPLORATION_DIR.parent  # modality_routing_classifier/
sys.path.append(str(CLASSIFIER_DIR))

from modality_data import load_jsonl  # noqa: E402
from modality_eval_metrics import compute_classification_metrics  # noqa: E402
from modality_label_mapping import LABEL_TO_ID, MODALITY_LABELS  # noqa: E402

DATA_DIR = CLASSIFIER_DIR / "exported_modality_routing_dataset"
RUNS_DIR = EXPLORATION_DIR / "runs"

# Hub revisions the reported numbers were produced with. Both models load custom
# code or custom weights from the Hub, so they are pinned instead of following main.
SCX_MODEL_ID = "scx-admin/scx-router-v0.1"
SCX_REVISION = "b45625de43a3bac2861d3f11b96c15a93f4a026e"
LFM25_MODEL_ID = "LiquidAI/LFM2.5-Encoder-350M"
LFM25_REVISION = "b886781f7c6f10ca9b7096e21b83e30a073c2f39"

LABEL_DESCRIPTIONS = {
    "AR": "a text-only response",
    "DIFFUSION": "generating an image",
    "BOTH": "both a text explanation and an image",
}


def label2description() -> dict[str, str]:
    """Return the label descriptions in the "NAME: description" form SCX expects.

    Returns:
        Description per label name.
    """
    return {name: f"{name}: {desc}" for name, desc in LABEL_DESCRIPTIONS.items()}


def pick_device(requested: str | None) -> str:
    """Choose the device to run on.

    Args:
        requested: An explicit device such as "cuda:0" or "cpu", or None to pick.

    Returns:
        The requested device, else "cuda" if available, else "cpu".
    """
    if requested:
        return requested
    import torch  # noqa: PLC0415  (lazy: the pure helpers here do not need torch)

    return "cuda" if torch.cuda.is_available() else "cpu"


def to_gliclass_examples(rows: list[dict]) -> list[dict]:
    """Convert dataset rows into GLiClass examples with one true label each.

    Args:
        rows: Rows with "text" and "label_name".

    Returns:
        Examples with "text", "all_labels" and "true_labels".
    """
    return [
        {
            "text": row["text"],
            "all_labels": list(MODALITY_LABELS),
            "true_labels": [row["label_name"]],
        }
        for row in rows
    ]


def make_gliclass_dataset(rows: list[dict], tokenizer, *, shuffle_labels: bool = True):
    """Build a GLiClass dataset in the multi-label form the SCX checkpoint supports.

    The decoder-KV architecture only implements a working loss for its native
    "multi_label_classification" problem type, where labels are a multi-hot vector.
    Its single-label path is broken as shipped, so the mutually exclusive 3-class
    task is expressed as multi-hot with exactly one active label per row.

    Args:
        rows: Rows with "text" and "label_name".
        tokenizer: The SCX tokenizer.
        shuffle_labels: Whether to shuffle label order per example; on for training,
            off for evaluation.

    Returns:
        The GLiClassDataset.
    """
    from gliclass.data_processing import (  # noqa: PLC0415  (lazy: heavy, SCX only)
        AugmentationConfig,
        GLiClassDataset,
    )

    return GLiClassDataset(
        to_gliclass_examples(rows),
        tokenizer,
        AugmentationConfig(enabled=False),
        label2description=label2description(),
        max_length=512,
        problem_type="multi_label_classification",
        architecture_type="decoder-kv",
        shuffle_labels=shuffle_labels,
    )


def summarize_predictions(truth: list[str], preds: list[str]) -> dict:
    """Score predicted label names against the true ones.

    Args:
        truth: True label names.
        preds: Predicted label names, aligned with truth.

    Returns:
        The classification metrics of modality_eval_metrics, plus "num_correct" and
        "num_rows".

    Raises:
        KeyError: If a label is not AR, DIFFUSION or BOTH.
    """
    y_true = np.array([LABEL_TO_ID[t] for t in truth], dtype=np.int64)
    y_pred = np.array([LABEL_TO_ID[p] for p in preds], dtype=np.int64)
    metrics = compute_classification_metrics(y_true, y_pred)
    metrics["num_correct"] = int((y_true == y_pred).sum())
    metrics["num_rows"] = len(truth)
    return metrics


def format_summary(model_name: str, metrics: dict) -> list[str]:
    """Format a summary from summarize_predictions for printing.

    Args:
        model_name: Name to show for the model.
        metrics: Result of summarize_predictions.

    Returns:
        The lines to print.
    """
    lines = [
        f"MODEL: {model_name}",
        f"ACCURACY: {metrics['accuracy']:.4f} ({metrics['num_correct']}/{metrics['num_rows']})",
        "CONFUSION (rows=true, cols=pred, order AR/DIFFUSION/BOTH):",
    ]
    lines += [
        f"  {label:10s} {row}"
        for label, row in zip(MODALITY_LABELS, metrics["confusion_matrix"], strict=True)
    ]
    for label in MODALITY_LABELS:
        scores = metrics["per_class"][label]
        lines.append(
            f"  {label:10s} precision={scores['precision']:.4f} recall={scores['recall']:.4f}"
        )
    return lines


def save_predictions(
    path: Path, model_name: str, accuracy: float, preds: list[str], rows: list[dict]
) -> None:
    """Write predictions in the format `judge_labels.py report --preds` reads.

    The file records the sha256 of every prompt, because the report tool refuses
    predictions it cannot tie to the rows it scores them against.

    Args:
        path: Output JSON path; parent directories are created.
        model_name: Name to record for the model.
        accuracy: Accuracy on the test split.
        preds: Predicted label names, aligned with rows.
        rows: The rows the predictions were made for, with "text".
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model_name,
        "accuracy": accuracy,
        "preds": preds,
        "input_hashes": [
            hashlib.sha256(row["text"].encode("utf-8")).hexdigest() for row in rows
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


__all__ = [
    "DATA_DIR",
    "LABEL_DESCRIPTIONS",
    "LFM25_MODEL_ID",
    "LFM25_REVISION",
    "MODALITY_LABELS",
    "RUNS_DIR",
    "SCX_MODEL_ID",
    "SCX_REVISION",
    "format_summary",
    "label2description",
    "load_jsonl",
    "make_gliclass_dataset",
    "pick_device",
    "save_predictions",
    "summarize_predictions",
    "to_gliclass_examples",
]
