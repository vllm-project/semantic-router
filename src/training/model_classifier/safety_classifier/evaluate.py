"""Evaluate Level-1/Level-2 safety artifacts on prepared JSONL data.

PyTorch, Transformers, and PEFT are imported only inside model-loading and
inference functions. Consequently JSONL parsing, CLI help, and unit tests for
the orchestration helpers work in a dependency-light environment.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from .config import DEFAULT_CONTRACT_PATH, id2label, load_contract, task_contract
    from .evaluation_io import (
        _label_value,
        _prediction_record,
        _sample_identity,
        _strict_single_target,
        _write_outputs,
        read_prepared_jsonl,
    )
    from .metrics import (
        binary_classification_metrics,
        freeze_binary_threshold,
        multiclass_classification_metrics,
        predictions_from_scores,
        probabilities_from_logits,
    )
except ImportError:  # Support ``python path/to/evaluate.py``.
    from evaluation_io import (
        _label_value,
        _prediction_record,
        _sample_identity,
        _strict_single_target,
        _write_outputs,
        read_prepared_jsonl,
    )
    from metrics import (
        binary_classification_metrics,
        freeze_binary_threshold,
        multiclass_classification_metrics,
        predictions_from_scores,
        probabilities_from_logits,
    )

    from config import DEFAULT_CONTRACT_PATH, id2label, load_contract, task_contract


@dataclass
class ModelBundle:
    """Loaded inference objects plus resolved artifact metadata."""

    model: Any
    tokenizer: Any
    device: Any
    artifact_type: str
    num_labels: int
    max_length: int


def detect_local_artifact_type(model_reference: str | Path) -> str | None:
    """Identify a local PEFT adapter or merged model; return ``None`` remotely."""
    path = Path(model_reference).expanduser()
    if not path.exists():
        return None
    if not path.is_dir():
        raise ValueError(f"model reference is not a directory: {path}")
    if (path / "adapter_config.json").is_file():
        return "adapter"
    if (path / "config.json").is_file():
        return "merged"
    raise ValueError(f"no adapter_config.json or config.json found in {path}")


def _resolve_artifact_type(
    model_reference: str,
    requested_type: str,
    *,
    revision: str | None,
) -> tuple[str, Any | None]:
    if requested_type not in {"auto", "adapter", "merged"}:
        raise ValueError("artifact_type must be auto, adapter, or merged")
    if requested_type != "auto":
        return requested_type, None

    local_type = detect_local_artifact_type(model_reference)
    if local_type is not None:
        return local_type, None

    # Repository names are a cheap and deterministic fast path for the release
    # contract. Unknown remote repositories are probed through PEFT metadata.
    lowered = model_reference.casefold()
    if lowered.endswith(("-lora", "-adapter")):
        return "adapter", None
    try:
        from peft import PeftConfig  # noqa: PLC0415
    except ImportError:
        return "merged", None
    try:
        peft_config = PeftConfig.from_pretrained(model_reference, revision=revision)
    except (OSError, ValueError):
        return "merged", None
    return "adapter", peft_config


def _load_tokenizer(
    auto_tokenizer: Any,
    sources: tuple[tuple[str, str | None], ...],
    *,
    trust_remote_code: bool,
) -> Any:
    tokenizer_error: OSError | None = None
    for source, revision in sources:
        try:
            return auto_tokenizer.from_pretrained(
                source,
                revision=revision,
                trust_remote_code=trust_remote_code,
            )
        except OSError as exc:
            tokenizer_error = exc
    raise RuntimeError(
        "unable to load a tokenizer for the evaluation artifact"
    ) from tokenizer_error


def _resolve_device(torch_module: Any, requested: str) -> Any:
    if requested == "auto":
        requested = "cuda" if torch_module.cuda.is_available() else "cpu"
    return torch_module.device(requested)


def load_model_bundle(
    model_reference: str,
    *,
    task_name: str,
    contract: dict[str, Any],
    artifact_type: str = "auto",
    model_revision: str | None = None,
    base_model: str | None = None,
    base_revision: str | None = None,
    device: str = "auto",
    trust_remote_code: bool = False,
) -> ModelBundle:
    """Load an adapter or merged sequence classifier with pinned task labels."""
    try:
        import torch  # noqa: PLC0415
        from transformers import (  # noqa: PLC0415
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )
    except ImportError as exc:
        raise RuntimeError(
            "evaluation requires torch and transformers; install requirements.txt"
        ) from exc

    task = task_contract(contract, task_name)
    labels_by_id = id2label(task)
    label2id = task["label2id"]
    resolved_type, probed_peft_config = _resolve_artifact_type(
        model_reference,
        artifact_type,
        revision=model_revision,
    )

    common_model_kwargs: dict[str, Any] = {
        "num_labels": task["num_labels"],
        "id2label": labels_by_id,
        "label2id": label2id,
        "trust_remote_code": trust_remote_code,
    }
    if resolved_type == "adapter":
        try:
            from peft import PeftConfig, PeftModel  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError("adapter evaluation requires peft") from exc
        peft_config = probed_peft_config or PeftConfig.from_pretrained(
            model_reference,
            revision=model_revision,
        )
        resolved_base_model = (
            base_model
            or contract["base_model"]["id"]
            or peft_config.base_model_name_or_path
        )
        resolved_base_revision = base_revision or contract["base_model"].get("revision")
        base = AutoModelForSequenceClassification.from_pretrained(
            resolved_base_model,
            revision=resolved_base_revision,
            **common_model_kwargs,
        )
        model = PeftModel.from_pretrained(
            base,
            model_reference,
            revision=model_revision,
        )
        tokenizer_sources = (
            (model_reference, model_revision),
            (resolved_base_model, resolved_base_revision),
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_reference,
            revision=model_revision,
            **common_model_kwargs,
        )
        tokenizer_sources = ((model_reference, model_revision),)

    tokenizer = _load_tokenizer(
        AutoTokenizer,
        tokenizer_sources,
        trust_remote_code=trust_remote_code,
    )
    resolved_device = _resolve_device(torch, device)
    model.to(resolved_device)
    model.eval()
    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        device=resolved_device,
        artifact_type=resolved_type,
        num_labels=int(task["num_labels"]),
        max_length=int(contract["model"]["max_length"]),
    )


def release_model_bundle(bundle: ModelBundle) -> None:
    """Release a sequential smoke model before loading the next one."""
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return
    del bundle.model
    del bundle.tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def predict_logits(
    bundle: ModelBundle,
    texts: list[str],
    *,
    batch_size: int,
) -> list[list[float]]:
    """Run deterministic, ordered batched inference and return CPU logits."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not texts:
        return []
    try:
        import torch  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("inference requires torch") from exc

    logits: list[list[float]] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = bundle.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=bundle.max_length,
                return_tensors="pt",
            )
            inputs = {name: tensor.to(bundle.device) for name, tensor in inputs.items()}
            output = bundle.model(**inputs).logits.detach().float().cpu().tolist()
            if any(len(row) != bundle.num_labels for row in output):
                raise RuntimeError(
                    f"model emitted logits width other than {bundle.num_labels}"
                )
            logits.extend(output)
    return logits


def _validate_single_options(
    task_name: str,
    *,
    threshold: float | None,
    freeze_threshold: bool,
) -> None:
    if task_name not in {"level1", "level2"}:
        raise ValueError("task_name must be level1 or level2")
    if freeze_threshold and task_name != "level1":
        raise ValueError("threshold freezing is only valid for level1")
    if freeze_threshold and threshold is not None:
        raise ValueError("choose either an explicit threshold or --freeze-threshold")


def _single_task_report(
    task_name: str,
    labels: list[int],
    probabilities: list[list[float]],
    rows: list[dict[str, Any]],
    names: list[str],
    *,
    threshold: float | None,
    freeze_threshold: bool,
    minimum_recall: float,
) -> tuple[list[int], dict[str, Any], dict[str, Any] | None]:
    if task_name == "level1":
        unsafe_scores = [row[1] for row in probabilities]
        threshold_selection = (
            freeze_binary_threshold(
                labels,
                unsafe_scores,
                minimum_recall=minimum_recall,
            )
            if freeze_threshold
            else None
        )
        resolved_threshold = (
            float(threshold_selection["threshold"])
            if threshold_selection is not None
            else 0.5
            if threshold is None
            else float(threshold)
        )
        predictions = predictions_from_scores(unsafe_scores, resolved_threshold)
        report = binary_classification_metrics(
            labels,
            predictions,
            unsafe_scores=unsafe_scores,
            threshold=resolved_threshold,
        )
        return predictions, report, threshold_selection

    predictions = [
        max(range(len(row)), key=lambda class_id: row[class_id])
        for row in probabilities
    ]
    report = multiclass_classification_metrics(
        labels,
        predictions,
        class_scores=probabilities,
        class_names=names,
        strict_single_target_mask=[_strict_single_target(row) for row in rows],
    )
    return predictions, report, None


def _single_prediction_records(
    rows: list[dict[str, Any]],
    labels: list[int],
    predictions: list[int],
    probabilities: list[list[float]],
    names: list[str],
    *,
    task_name: str,
    include_text: bool,
) -> list[dict[str, Any]]:
    records = [
        _prediction_record(
            row,
            index,
            reference=reference,
            prediction=prediction,
            scores=scores,
            names=names,
            include_text=include_text,
        )
        for index, (row, reference, prediction, scores) in enumerate(
            zip(rows, labels, predictions, probabilities, strict=True)
        )
    ]
    if task_name == "level1":
        for record, scores in zip(records, probabilities, strict=True):
            record["unsafe_score"] = scores[1]
    return records


def evaluate_model(
    *,
    model_reference: str,
    data_path: str | Path,
    output_dir: str | Path,
    task_name: str,
    contract_path: str | Path = DEFAULT_CONTRACT_PATH,
    artifact_type: str = "auto",
    model_revision: str | None = None,
    base_model: str | None = None,
    base_revision: str | None = None,
    device: str = "auto",
    batch_size: int = 16,
    split: str | None = None,
    limit: int | None = None,
    threshold: float | None = None,
    freeze_threshold: bool = False,
    minimum_recall: float = 0.95,
    include_text: bool = False,
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    """Evaluate one artifact and write ``metrics.json``/``predictions.jsonl``."""
    _validate_single_options(
        task_name,
        threshold=threshold,
        freeze_threshold=freeze_threshold,
    )

    contract = load_contract(contract_path)
    task = task_contract(contract, task_name)
    names_by_id = id2label(task)
    names = [names_by_id[index] for index in range(task["num_labels"])]
    rows = read_prepared_jsonl(
        data_path,
        task_name=task_name,
        split=split,
        limit=limit,
    )
    labels = [_label_value(row, task_name, task["label2id"]) for row in rows]
    bundle = load_model_bundle(
        model_reference,
        task_name=task_name,
        contract=contract,
        artifact_type=artifact_type,
        model_revision=model_revision,
        base_model=base_model,
        base_revision=base_revision,
        device=device,
        trust_remote_code=trust_remote_code,
    )
    resolved_artifact_type = bundle.artifact_type
    try:
        logits = predict_logits(
            bundle, [row["text"] for row in rows], batch_size=batch_size
        )
    finally:
        release_model_bundle(bundle)
    probabilities = probabilities_from_logits(logits)
    predictions, report, threshold_selection = _single_task_report(
        task_name,
        labels,
        probabilities,
        rows,
        names,
        threshold=threshold,
        freeze_threshold=freeze_threshold,
        minimum_recall=minimum_recall,
    )
    prediction_records = _single_prediction_records(
        rows,
        labels,
        predictions,
        probabilities,
        names,
        task_name=task_name,
        include_text=include_text,
    )

    payload = {
        "schema_version": 1,
        "evaluation": {
            "mode": "single",
            "task": task_name,
            "model": model_reference,
            "model_revision": model_revision,
            "artifact_type": resolved_artifact_type,
            "data": str(Path(data_path)),
            "split": split,
            "num_examples": len(rows),
        },
        "threshold_selection": threshold_selection,
        "metrics": report,
    }
    _write_outputs(output_dir, payload, prediction_records)
    return payload


def _stage_probabilities(
    model_reference: str,
    texts: list[str],
    *,
    task_name: str,
    contract: dict[str, Any],
    artifact_type: str,
    model_revision: str | None,
    base_model: str | None,
    base_revision: str | None,
    device: str,
    batch_size: int,
    trust_remote_code: bool,
) -> tuple[str, list[list[float]]]:
    bundle = load_model_bundle(
        model_reference,
        task_name=task_name,
        contract=contract,
        artifact_type=artifact_type,
        model_revision=model_revision,
        base_model=base_model,
        base_revision=base_revision,
        device=device,
        trust_remote_code=trust_remote_code,
    )
    resolved_type = bundle.artifact_type
    try:
        logits = predict_logits(bundle, texts, batch_size=batch_size)
    finally:
        release_model_bundle(bundle)
    return resolved_type, probabilities_from_logits(logits)


def _hierarchical_prediction_records(
    rows: list[dict[str, Any]],
    level1_probabilities: list[list[float]],
    level1_predictions: list[int],
    level2_probabilities_by_index: dict[int, list[float]],
    level2_names_by_id: dict[int, str],
    *,
    include_text: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, (row, level1_scores, level1_prediction) in enumerate(
        zip(rows, level1_probabilities, level1_predictions, strict=True)
    ):
        level2_scores = level2_probabilities_by_index.get(index)
        level2_prediction = (
            max(range(len(level2_scores)), key=lambda class_id: level2_scores[class_id])
            if level2_scores is not None
            else None
        )
        record: dict[str, Any] = {
            "sample_id": _sample_identity(row, index),
            "text_sha256": hashlib.sha256(row["text"].encode("utf-8")).hexdigest(),
            "level1_prediction": level1_prediction,
            "level1_prediction_label": "unsafe" if level1_prediction else "safe",
            "level1_scores": level1_scores,
            "unsafe_score": level1_scores[1],
            "level2_prediction": level2_prediction,
            "level2_prediction_label": (
                level2_names_by_id[level2_prediction]
                if level2_prediction is not None
                else None
            ),
            "level2_scores": level2_scores,
            "final_prediction": (
                "safe"
                if level1_prediction == 0
                else level2_names_by_id[level2_prediction]
            ),
        }
        if include_text:
            record["text"] = row["text"]
        records.append(record)
    return records


def _hierarchical_payload(
    *,
    level1_model: str,
    level2_model: str,
    level1_revision: str | None,
    level2_revision: str | None,
    resolved_level1_type: str,
    resolved_level2_type: str,
    data_path: str | Path,
    split: str | None,
    count: int,
    routed_count: int,
    threshold: float,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "evaluation": {
            "mode": "hierarchical_smoke",
            "level1_model": level1_model,
            "level1_revision": level1_revision,
            "level1_artifact_type": resolved_level1_type,
            "level2_model": level2_model,
            "level2_revision": level2_revision,
            "level2_artifact_type": resolved_level2_type,
            "data": str(Path(data_path)),
            "split": split,
            "num_examples": count,
        },
        "metrics": {
            "smoke_passed": True,
            "examples": count,
            "routed_to_level2": routed_count,
            "level2_route_rate": routed_count / count,
            "level2_probe_only": routed_count == 0,
            "threshold": float(threshold),
            "threshold_comparison": "unsafe_score >= threshold",
        },
    }


def evaluate_hierarchical(
    *,
    level1_model: str,
    level2_model: str,
    data_path: str | Path,
    output_dir: str | Path,
    contract_path: str | Path = DEFAULT_CONTRACT_PATH,
    level1_artifact_type: str = "auto",
    level2_artifact_type: str = "auto",
    level1_revision: str | None = None,
    level2_revision: str | None = None,
    base_model: str | None = None,
    base_revision: str | None = None,
    device: str = "auto",
    batch_size: int = 16,
    split: str | None = None,
    limit: int | None = 8,
    threshold: float = 0.5,
    include_text: bool = False,
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    """Smoke the hierarchy: Level-2 runs only for Level-1 unsafe examples."""
    contract = load_contract(contract_path)
    rows = read_prepared_jsonl(data_path, split=split, limit=limit)
    texts = [row["text"] for row in rows]

    resolved_level1_type, level1_probabilities = _stage_probabilities(
        level1_model,
        texts,
        task_name="level1",
        contract=contract,
        artifact_type=level1_artifact_type,
        model_revision=level1_revision,
        base_model=base_model,
        base_revision=base_revision,
        device=device,
        batch_size=batch_size,
        trust_remote_code=trust_remote_code,
    )
    unsafe_scores = [row[1] for row in level1_probabilities]
    level1_predictions = predictions_from_scores(unsafe_scores, threshold)

    unsafe_indices = [
        index for index, prediction in enumerate(level1_predictions) if prediction == 1
    ]
    level2_probabilities_by_index: dict[int, list[float]] = {}
    # A smoke must load and execute both artifacts. If Level-1 routes no input,
    # execute Level-2 on one probe but do not treat that probe as a real route.
    level2_inference_indices = unsafe_indices or [0]
    resolved_level2_type, level2_probabilities = _stage_probabilities(
        level2_model,
        [texts[index] for index in level2_inference_indices],
        task_name="level2",
        contract=contract,
        artifact_type=level2_artifact_type,
        model_revision=level2_revision,
        base_model=base_model,
        base_revision=base_revision,
        device=device,
        batch_size=batch_size,
        trust_remote_code=trust_remote_code,
    )
    if unsafe_indices:
        level2_probabilities_by_index = dict(
            zip(unsafe_indices, level2_probabilities, strict=True)
        )

    level2_task = task_contract(contract, "level2")
    level2_names_by_id = id2label(level2_task)
    prediction_records = _hierarchical_prediction_records(
        rows,
        level1_probabilities,
        level1_predictions,
        level2_probabilities_by_index,
        level2_names_by_id,
        include_text=include_text,
    )
    count = len(rows)
    payload = _hierarchical_payload(
        level1_model=level1_model,
        level2_model=level2_model,
        level1_revision=level1_revision,
        level2_revision=level2_revision,
        resolved_level1_type=resolved_level1_type,
        resolved_level2_type=resolved_level2_type,
        data_path=data_path,
        split=split,
        count=count,
        routed_count=len(unsafe_indices),
        threshold=threshold,
    )
    _write_outputs(output_dir, payload, prediction_records)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Prepared JSONL input")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT_PATH))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--split")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--include-text", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--base-model")
    parser.add_argument("--base-revision")

    parser.add_argument("--task", choices=("level1", "level2"))
    parser.add_argument("--model")
    parser.add_argument(
        "--artifact-type", choices=("auto", "adapter", "merged"), default="auto"
    )
    parser.add_argument("--model-revision")
    parser.add_argument("--freeze-threshold", action="store_true")
    parser.add_argument("--minimum-recall", type=float, default=0.95)

    parser.add_argument("--hierarchical-smoke", action="store_true")
    parser.add_argument("--level1-model")
    parser.add_argument("--level2-model")
    parser.add_argument(
        "--level1-artifact-type",
        choices=("auto", "adapter", "merged"),
        default="auto",
    )
    parser.add_argument(
        "--level2-artifact-type",
        choices=("auto", "adapter", "merged"),
        default="auto",
    )
    parser.add_argument("--level1-revision")
    parser.add_argument("--level2-revision")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    common = {
        "data_path": args.data,
        "output_dir": args.output_dir,
        "contract_path": args.contract,
        "base_model": args.base_model,
        "base_revision": args.base_revision,
        "device": args.device,
        "batch_size": args.batch_size,
        "split": args.split,
        "include_text": args.include_text,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.hierarchical_smoke:
        if not args.level1_model or not args.level2_model:
            raise SystemExit(
                "--hierarchical-smoke requires --level1-model and --level2-model"
            )
        evaluate_hierarchical(
            level1_model=args.level1_model,
            level2_model=args.level2_model,
            level1_artifact_type=args.level1_artifact_type,
            level2_artifact_type=args.level2_artifact_type,
            level1_revision=args.level1_revision,
            level2_revision=args.level2_revision,
            limit=8 if args.limit is None else args.limit,
            threshold=0.5 if args.threshold is None else args.threshold,
            **common,
        )
    else:
        if not args.task or not args.model:
            raise SystemExit("single evaluation requires --task and --model")
        evaluate_model(
            model_reference=args.model,
            task_name=args.task,
            artifact_type=args.artifact_type,
            model_revision=args.model_revision,
            limit=args.limit,
            threshold=args.threshold,
            freeze_threshold=args.freeze_threshold,
            minimum_recall=args.minimum_recall,
            **common,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
