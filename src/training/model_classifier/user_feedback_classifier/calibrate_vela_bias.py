"""Calibrate Vela Feedback's applicability bias on explicitly supplied development data.

Grid screening never substitutes for a real FP32 forward with the changed head.
The source checkpoint is read-only; rejected experiments retain their evidence.
"""

# Keep optional numerical dependencies out of data-only imports.
# ruff: noqa: PLC0415

import argparse
import hashlib
import json
import math
from pathlib import Path

from ..sequence_repair.data import classification_metrics, file_receipts, read_records
from ..sequence_repair.evaluate import evaluate_records
from ..sequence_repair.runtime_mapping import write_runtime_mappings
from .vela_contract import VELA_ID2LABEL, VELA_LABEL2ID

BIAS_GRID = tuple(index / 8 for index in range(9))
CONTEXT_BUCKETS = ("256", "4096", "8192", "16384", "32768")
NF_INDEX = VELA_LABEL2ID["NO_FEEDBACK"]
SOURCE_F1_MIN = 0.85
CONTEXT_F1_MIN = 0.8
NF_PRECISION_RECALL_MIN = 0.9
PROBABILITY_DIMENSIONS = 2


def gate_results(metrics, required_lengths=CONTEXT_BUCKETS):
    """Require actual source, class and length support; reject vacuous passes."""
    sources = metrics["breakdowns"]["source"]
    lengths = metrics["breakdowns"]["length_bucket"]
    nf = metrics["per_label"]["NO_FEEDBACK"]
    return {
        "class_support": all(
            metrics["per_label"][label]["support"] > 0 for label in VELA_LABEL2ID
        ),
        "sources": bool(sources)
        and "unspecified" not in sources
        and all(
            group["macro_f1_present_labels"] >= SOURCE_F1_MIN
            for group in sources.values()
        ),
        "lengths": bool(required_lengths)
        and all(bucket in lengths for bucket in required_lengths)
        and all(
            group["macro_f1"] >= CONTEXT_F1_MIN
            for bucket, group in lengths.items()
            if bucket.isdigit()
        ),
        "no_feedback_precision": nf["precision"] >= NF_PRECISION_RECALL_MIN,
        "no_feedback_recall": nf["recall"] >= NF_PRECISION_RECALL_MIN,
    }


def adjusted_probabilities(probabilities, bias):
    """Screen a nonnegative logit shift using the original FP32 probabilities."""
    import numpy as np

    if not math.isfinite(bias) or bias < 0 or bias > max(BIAS_GRID):
        raise ValueError("Bias must be finite and inside the frozen [0, 1] grid range")
    values = np.asarray(probabilities, dtype=np.float32)
    if (
        values.ndim != PROBABILITY_DIMENSIONS
        or values.shape[1] != len(VELA_LABEL2ID)
        or not len(values)
        or not np.isfinite(values).all()
        or (values < 0).any()
        or (values > 1).any()
        or not np.allclose(values.sum(-1), 1, atol=1e-6, rtol=0)
    ):
        raise ValueError("Expected finite, normalized five-class probabilities")
    result = values.copy()
    result[:, NF_INDEX] *= np.float32(np.exp(bias))
    result /= result.sum(-1, keepdims=True)
    return result


def probability_metrics(rows, probabilities):
    gold = [VELA_LABEL2ID[row["label"]] for row in rows]
    predicted = probabilities.argmax(-1).tolist()
    if len(gold) != len(predicted):
        raise ValueError("Probability rows differ from development records")
    metrics = classification_metrics(gold, predicted, VELA_ID2LABEL)
    metrics["breakdowns"] = {}
    for field in ("source", "length_bucket"):
        metrics["breakdowns"][field] = {}
        for group in sorted({str(row.get(field, "unspecified")) for row in rows}):
            indices = [
                index
                for index, row in enumerate(rows)
                if str(row.get(field, "unspecified")) == group
            ]
            metrics["breakdowns"][field][group] = classification_metrics(
                [gold[index] for index in indices],
                [predicted[index] for index in indices],
                VELA_ID2LABEL,
            )
    return metrics


def screen_grid(rows, probabilities, required_lengths=CONTEXT_BUCKETS):
    trials, selected = [], None
    for bias in BIAS_GRID:
        metrics = probability_metrics(rows, adjusted_probabilities(probabilities, bias))
        gates = gate_results(metrics, required_lengths)
        trials.append({"bias": bias, "gates": gates, "metrics": metrics})
        if selected is None and all(gates.values()):
            selected = bias
    return selected, trials


def apply_head_bias(model, bias):
    """Change exactly the applicability bias of a standard FP32 five-class head."""
    import torch

    if (
        model.config.label2id != VELA_LABEL2ID
        or {int(key): value for key, value in model.config.id2label.items()}
        != VELA_ID2LABEL
        or model.config.problem_type != "single_label_classification"
        or model.classifier.bias.shape != (len(VELA_LABEL2ID),)
        or model.classifier.bias.dtype != torch.float32
        or bias not in BIAS_GRID
    ):
        raise ValueError("Expected the exact FP32 Vela five-class head and frozen grid")
    before = model.classifier.bias.detach().clone()
    with torch.no_grad():
        model.classifier.bias[NF_INDEX].add_(bias)
    return before.cpu().tolist(), model.classifier.bias.detach().cpu().tolist()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def save_evaluation(directory, name, metrics, predictions):
    write_json(directory / f"{name}.json", metrics)
    (directory / f"{name}.predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in predictions)
    )


def main():
    import torch
    import transformers
    from safetensors.torch import load_file
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--development", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--required-lengths", nargs="+", default=list(CONTEXT_BUCKETS))
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite calibration evidence or a checkpoint")
    source = args.model / "model.safetensors"
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    torch.set_num_threads(8)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, torch_dtype=torch.float32, attn_implementation="sdpa"
    ).to(args.device)
    if (
        type(model).__name__ != "ModernBertForSequenceClassification"
        or model.config.label2id != VELA_LABEL2ID
        or not 0 < args.max_length <= model.config.max_position_embeddings
    ):
        raise ValueError(
            "Expected a Vela Feedback checkpoint and supported input budget"
        )
    rows = read_records(args.development, VELA_LABEL2ID)
    args.output.mkdir(parents=True)
    metrics, predictions = evaluate_records(
        model,
        tokenizer,
        rows,
        VELA_LABEL2ID,
        VELA_ID2LABEL,
        args.max_length,
        dtype="float32",
    )
    save_evaluation(args.output, "original", metrics, predictions)
    selected, trials = screen_grid(
        rows, [row["probabilities"] for row in predictions], args.required_lengths
    )
    receipt = {
        "grid": list(BIAS_GRID),
        "selected_bias": selected,
        "selection": "Smallest feasible bias; no grid expansion or final-data selection",
        "development": file_receipts(args.development),
        "required_lengths": args.required_lengths,
        "source_weights_sha256": source_hash,
        "source_config_sha256": hashlib.sha256(
            (args.model / "config.json").read_bytes()
        ).hexdigest(),
        "implementation_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "transformers": transformers.__version__,
        "forward_dtype": "float32",
        "model_input": "current user turn only",
        "trials": trials,
    }
    write_json(args.output / "calibration.json", receipt)
    if selected is None:
        raise ValueError("No frozen-grid candidate satisfies every development gate")
    before, after = apply_head_bias(model, selected)
    actual, actual_predictions = evaluate_records(
        model,
        tokenizer,
        rows,
        VELA_LABEL2ID,
        VELA_ID2LABEL,
        args.max_length,
        dtype="float32",
    )
    save_evaluation(args.output, "calibrated-actual", actual, actual_predictions)
    gates = gate_results(actual, args.required_lengths)
    if not all(gates.values()):
        raise ValueError("Actual changed-head forward failed; no checkpoint exported")
    original = load_file(source, device="cpu")
    changed = [
        key
        for key, value in model.state_dict().items()
        if not torch.equal(value.cpu(), original[key])
    ]
    expected_changes = ["classifier.bias"] if selected else []
    if changed != expected_changes:
        raise ValueError("Calibration changed unexpected model tensors")
    if hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
        raise ValueError("Source weights changed during calibration")
    destination = args.output / "model"
    model.cpu().save_pretrained(destination, safe_serialization=True)
    tokenizer.save_pretrained(destination)
    write_runtime_mappings(destination, VELA_LABEL2ID, "feedback")
    receipt.update(
        actual_gates=gates,
        original_bias=before,
        calibrated_bias=after,
        changed_tensors=changed,
        output_weights_sha256=hashlib.sha256(
            (destination / "model.safetensors").read_bytes()
        ).hexdigest(),
        independent_generalization_established=False,
    )
    write_json(args.output / "calibration.json", receipt)
    print(json.dumps({"selected_bias": selected, "actual_gates": gates}))


if __name__ == "__main__":
    main()
