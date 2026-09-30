#!/usr/bin/env python3
"""Run the pinned Semantic Router domain classifier on JSONL cases.

This is an evaluation runner, not a Router runtime path. It loads the exact
Vela Domain artifact used by the maintained configuration, preserves the
artifact's complete 14-class distribution, and emits one result per input.
The default revision is pinned in ``baseline.yaml``; pass ``--model-dir`` to
use an already downloaded snapshot without changing that identity.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "baseline.yaml"
DEFAULT_MAX_LENGTH = 32768
DISTRIBUTION_SUM_TOLERANCE = 1e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mapping", type=Path, default=None)
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Existing model snapshot; otherwise download the pinned revision",
    )
    parser.add_argument("--model-repo", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Unrecorded model calls before timing cases (default: 1)",
    )
    parser.add_argument("--warmup-output", type=Path, default=None)
    parser.add_argument("--metadata-output", type=Path, default=None)
    parser.add_argument(
        "--mapping-only",
        action="store_true",
        help="Validate and print the pinned mapping without loading model weights",
    )
    return parser.parse_args()


def load_mapping(path: Path) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mapping = payload.get("category_to_idx")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(f"{path} has no category_to_idx mapping")
    normalized = {str(label): int(index) for label, index in mapping.items()}
    if sorted(normalized.values()) != list(range(len(normalized))):
        raise ValueError(f"{path} does not contain contiguous class IDs")
    reverse = payload.get("idx_to_category")
    expected_reverse = {str(index): label for label, index in normalized.items()}
    if reverse != expected_reverse:
        raise ValueError(f"{path} has inconsistent forward and reverse mappings")
    return normalized


def load_eval_config(path: Path) -> dict[str, Any]:
    try:
        import yaml  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("PyYAML is required to read baseline.yaml") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    model = payload.get("model") if isinstance(payload, dict) else None
    if not isinstance(model, dict):
        raise ValueError(f"{path} has no model configuration")
    for key in ("repo", "revision", "mapping_file"):
        if not model.get(key):
            raise ValueError(f"{path} model configuration is missing {key!r}")
    return payload


def load_cases(path: Path, mapping: dict[str, int]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        case_id = str(row.get("case_id", ""))
        text = row.get("text")
        if not case_id or case_id in seen:
            raise ValueError(f"{path}:{line_number}: case_id must be unique")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{path}:{line_number}: text must be non-empty")
        expected = row.get("expected_label")
        if expected is not None and expected not in mapping:
            raise ValueError(
                f"{path}:{line_number}: unknown expected_label {expected!r}"
            )
        seen.add(case_id)
        cases.append({"case_id": case_id, "text": text, "expected_label": expected})
    if not cases:
        raise ValueError(f"{path}: no cases")
    return cases


def resolve_model_dir(args: argparse.Namespace) -> tuple[Path, str]:
    if args.model_dir is not None:
        if not args.model_dir.is_dir():
            raise FileNotFoundError(args.model_dir)
        return args.model_dir, args.revision
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    model_dir = snapshot_download(
        repo_id=args.model_repo,
        revision=args.revision,
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"],
        max_workers=1,
    )
    return Path(model_dir), args.revision


def _sync(device: str) -> None:
    import torch  # noqa: PLC0415

    if device == "cuda":
        torch.cuda.synchronize()


def run(args: argparse.Namespace) -> None:
    run_started = datetime.now(timezone.utc).isoformat()
    eval_config = load_eval_config(args.config)
    model_config = eval_config["model"]
    args.model_repo = args.model_repo or model_config["repo"]
    args.revision = args.revision or model_config["revision"]
    args.mapping = args.mapping or args.config.parent / model_config["mapping_file"]
    mapping = load_mapping(args.mapping)
    cases = load_cases(args.input, mapping)
    if args.mapping_only:
        print(json.dumps({"labels": mapping, "class_count": len(mapping)}, indent=2))
        return

    import torch  # noqa: PLC0415
    from transformers import (  # noqa: PLC0415
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_dir, resolved_revision = resolve_model_dir(args)
    artifact_mapping_path = model_dir / "category_mapping.json"
    if artifact_mapping_path.is_file():
        artifact_mapping = load_mapping(artifact_mapping_path)
        if artifact_mapping != mapping:
            raise ValueError(
                "downloaded artifact category_mapping.json differs from the "
                "pinned evaluation mapping"
            )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = (
        AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    )
    if int(model.config.num_labels) != len(mapping):
        raise ValueError(
            f"model has {model.config.num_labels} labels, mapping has {len(mapping)}"
        )
    config_id2label = getattr(model.config, "id2label", {})
    if config_id2label and all(str(i) in config_id2label for i in range(len(mapping))):
        declared = {str(config_id2label[str(i)]) for i in range(len(mapping))}
        if not declared.intersection(mapping):
            raise ValueError("model config labels do not match category_mapping.json")

    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs cannot be negative")
    if args.warmup_runs:
        warmup = tokenizer(cases[0]["text"], return_tensors="pt", truncation=False)
        if int(warmup["attention_mask"].sum().item()) > args.max_length:
            raise ValueError("warm-up input exceeds --max-length")
        warmup = {key: value.to(device) for key, value in warmup.items()}
        warmup_record = None
        with torch.inference_mode():
            for _ in range(args.warmup_runs):
                warmup_logits = model(**warmup).logits
        if args.warmup_output is not None and args.warmup_runs:
            warmup_probs = torch.softmax(warmup_logits[0].float(), dim=-1).cpu().tolist()
            warmup_by_label = {label: warmup_probs[index] for label, index in sorted(mapping.items(), key=lambda item: item[1])}
            warmup_record = {"record_type":"warmup","case_id":"warmup-001","text":cases[0]["text"],"warmup_runs":args.warmup_runs,"prediction":max(warmup_by_label,key=warmup_by_label.get),"probabilities":warmup_by_label,"contract_valid":True,"device":device,"model_repo":args.model_repo,"model_revision":resolved_revision,"mapping_sha256":hashlib.sha256(args.mapping.read_bytes()).hexdigest()}
            args.warmup_output.parent.mkdir(parents=True, exist_ok=True)
            args.warmup_output.write_text(json.dumps(warmup_record, ensure_ascii=False, sort_keys=True)+"\n")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mapping_digest = hashlib.sha256(args.mapping.read_bytes()).hexdigest()
    with args.output.open("w", encoding="utf-8") as output:
        for case in cases:
            result: dict[str, Any] = {
                "case_id": case["case_id"],
                "expected_label": case["expected_label"],
                "model_repo": args.model_repo,
                "model_revision": resolved_revision,
                "mapping_sha256": mapping_digest,
                "contract": "label_distribution.v1",
                "device": device,
                "max_length": args.max_length,
                "warmup_runs": args.warmup_runs,
                "attempts": 1,
                "error": None,
            }
            try:
                encoded = tokenizer(
                    case["text"],
                    return_tensors="pt",
                    truncation=False,
                )
                token_count = int(encoded["attention_mask"].sum().item())
                if token_count > args.max_length:
                    raise ValueError(
                        f"input has {token_count} tokens, max is {args.max_length}"
                    )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                _sync(device)
                wall_start = time.perf_counter()
                model_start = time.perf_counter()
                with torch.inference_mode():
                    logits = model(**encoded).logits
                _sync(device)
                model_ms = (time.perf_counter() - model_start) * 1000
                wall_ms = (time.perf_counter() - wall_start) * 1000
                probabilities = torch.softmax(logits[0].float(), dim=-1).cpu().tolist()
                if len(probabilities) != len(mapping) or not all(
                    math.isfinite(value) and 0 <= value <= 1 for value in probabilities
                ):
                    raise ValueError(
                        "model returned a non-finite or invalid distribution"
                    )
                probability_sum = sum(probabilities)
                if abs(probability_sum - 1.0) > DISTRIBUTION_SUM_TOLERANCE:
                    raise ValueError(f"probabilities sum to {probability_sum}, not 1")
                by_label = {
                    label: probabilities[index]
                    for label, index in sorted(
                        mapping.items(), key=lambda item: item[1]
                    )
                }
                prediction = max(by_label, key=by_label.get)
                result.update(
                    prediction=prediction,
                    probabilities=by_label,
                    contract_valid=True,
                    token_count=token_count,
                    latency_wall_ms=wall_ms,
                    latency_model_ms=model_ms,
                )
                if case["expected_label"] is not None:
                    result["correct"] = prediction == case["expected_label"]
            except Exception as exc:  # preserve one explicit failure record per case
                result.update(prediction=None, probabilities=None, contract_valid=False)
                result["error"] = f"{type(exc).__name__}: {exc}"
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
    run_finished = datetime.now(timezone.utc).isoformat()
    if args.metadata_output is not None:
        args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_output.write_text(json.dumps({"run_started":run_started,"run_finished":run_finished,"device":device,"warmup_runs":args.warmup_runs,"attempts_per_case":1,"automatic_retries":False}, indent=2)+"\n")


if __name__ == "__main__":
    try:
        run(parse_args())
    except (OSError, ValueError, ImportError) as exc:
        print(f"baseline runner: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
