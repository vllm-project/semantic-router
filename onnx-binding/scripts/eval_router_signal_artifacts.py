#!/usr/bin/env python3
"""Evaluate router-owned ONNX signal artifacts across runtime/provider paths.

This harness compares the migration matrix used for AMD/MIGraphX work:

* old runtime + old ONNX artifact
* new runtime architecture + old ONNX artifact
* new runtime architecture + new ONNX artifact on CPU
* new runtime architecture + new ONNX artifact on AMD/auto providers

It intentionally uses direct ONNX Runtime sessions so each run can force an
exact ONNX file. That makes artifact quality and provider drift visible before
the same matrix is promoted into the Rust/Go runtime path.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import statistics
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT_PARENT_INDEX = 2
SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = (
    SCRIPT_PATH.parents[REPO_ROOT_PARENT_INDEX]
    if len(SCRIPT_PATH.parents) > REPO_ROOT_PARENT_INDEX
    and (
        SCRIPT_PATH.parents[REPO_ROOT_PARENT_INDEX] / "e2e/testcases/testdata"
    ).exists()
    else Path.cwd()
)
MAX_CLASSIFICATION_SEQ_LEN = 512
HF_SPEC_SPLIT_PARTS = 3
HF_SPEC_CONFIG_SPLIT_PARTS = 4
HF_VIEWER_ROWS_PAGE_SIZE = 100
RUN_SPEC_ARTIFACT_PROVIDER_PARTS = 2
RUN_SPEC_RUNTIME_PARTS = 3
SEQUENCE_LOGITS_RANK = 2
TOKEN_LOGITS_RANK = 3
MISMATCH_SAMPLE_LIMIT = 20
MODULE_CACHE: dict[str, Any] = {}
NISQ_DATASET_URL = (
    "https://raw.githubusercontent.com/YaoSun0422/NISQ_dataset/main/final_train.csv"
)


def require_onnxruntime():
    if "onnxruntime" in MODULE_CACHE:
        return MODULE_CACHE["onnxruntime"]
    try:
        imported_ort = importlib.import_module("onnxruntime")
    except ImportError as exc:  # pragma: no cover - exercised by operator envs.
        raise SystemExit(
            "onnxruntime is required. Install CPU onnxruntime locally or AMD's "
            "onnxruntime_migraphx wheel in the ROCm validation image."
        ) from exc
    MODULE_CACHE["onnxruntime"] = imported_ort
    return imported_ort


def require_transformers():
    if "transformers" in MODULE_CACHE:
        module = MODULE_CACHE["transformers"]
        return module.AutoTokenizer, module.PreTrainedTokenizerFast
    try:
        module = importlib.import_module("transformers")
    except ImportError as exc:  # pragma: no cover - exercised by operator envs.
        raise SystemExit(
            "transformers is required for tokenizer-compatible eval."
        ) from exc
    MODULE_CACHE["transformers"] = module
    return module.AutoTokenizer, module.PreTrainedTokenizerFast


@dataclass(frozen=True)
class EvalExample:
    id: str
    text: str
    label: str | int | None
    entities: tuple[dict[str, Any], ...]
    source: str
    raw: dict[str, Any]


@dataclass(frozen=True)
class RunSpec:
    name: str
    artifact: Path
    provider: str
    runtime: str


@dataclass
class RunResult:
    spec: RunSpec
    status: str
    selected_providers: list[str]
    fallback_reason: str | None
    error: str | None
    predictions: list[dict[str, Any]]
    latency_ms: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate old/new ONNX signal artifacts across providers."
    )
    parser.add_argument(
        "--task",
        required=True,
        help="Task name for reports, e.g. intent, feedback, factcheck, jailbreak, pii.",
    )
    parser.add_argument(
        "--task-type",
        choices=("sequence", "token"),
        required=True,
        help="sequence for classifiers, token for PII/BIO token classifiers.",
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        type=Path,
        help="Model directory containing config/tokenizer and ONNX artifacts.",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=None,
        help="Optional tokenizer directory/file source. Defaults to --model-dir.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help=(
            "Dataset spec. Supported: json:/path, jsonl:/path, "
            "hf:repo[:split], hf:repo:config:split, e2e-domain, e2e-pii, "
            "e2e-jailbreak, factcheck-nisq, jailbreak-toxic-chat, "
            "jailbreak-salad, jailbreak-mixed."
        ),
    )
    parser.add_argument(
        "--text-field",
        default=None,
        help="Input text field override for JSON/HF datasets.",
    )
    parser.add_argument(
        "--label-field",
        default=None,
        help="Gold label field override for sequence JSON/HF datasets.",
    )
    parser.add_argument(
        "--entities-field",
        default="entities",
        help="Gold entity field for token JSON/HF datasets.",
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help=(
            "Run spec NAME=ARTIFACT:PROVIDER[:RUNTIME]. ARTIFACT may be absolute "
            "or relative to --model-dir or --model-dir/onnx. Providers: cpu, "
            "migraphx, rocm, auto."
        ),
    )
    parser.add_argument(
        "--baseline-run",
        default=None,
        help="Run name used as quality baseline. Defaults to the first --run.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=MAX_CLASSIFICATION_SEQ_LEN)
    parser.add_argument(
        "--pad-to-max-length",
        action="store_true",
        help=(
            "Pad every request to --max-length. This keeps MIGraphX input shapes "
            "stable during quality eval and avoids measuring per-shape cold compile."
        ),
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional sample limit.")
    parser.add_argument("--jsonl", type=Path, default=None)
    parser.add_argument("--summary-md", type=Path, default=None)
    parser.add_argument(
        "--positive-label",
        action="append",
        default=[],
        help="Safety-critical positive labels whose recall must not regress.",
    )
    parser.add_argument("--min-label-match", type=float, default=0.999)
    parser.add_argument("--max-accuracy-drop", type=float, default=0.001)
    parser.add_argument("--max-macro-f1-drop", type=float, default=0.001)
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit non-zero if configured regression gates fail.",
    )
    return parser.parse_args()


def read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                rows.append(json.loads(line))
        return rows

    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("examples", "test_cases", "cases", "data", "rows"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(f"Unsupported JSON dataset shape in {path}")


def load_hf_rows(spec: str, limit: int) -> list[dict[str, Any]]:
    parts = spec.split(":")
    dataset_id = parts[1]
    config = "default"
    split = "validation"
    if len(parts) == HF_SPEC_SPLIT_PARTS:
        split = parts[2]
    elif len(parts) == HF_SPEC_CONFIG_SPLIT_PARTS:
        config = parts[2]
        split = parts[3]

    return load_hf_config_rows(dataset_id, split, limit, config=config)


def load_hf_config_rows(
    dataset_id: str, split: str, limit: int, *, config: str = "default"
) -> list[dict[str, Any]]:
    try:
        return load_hf_viewer_rows(dataset_id, split, limit, config=config)
    except Exception as viewer_exc:
        viewer_error = viewer_exc
    try:
        datasets_module = importlib.import_module("datasets")
    except ImportError as exc:
        raise RuntimeError(
            f"HF Dataset Viewer failed for {dataset_id}:{split}: {viewer_error}"
        ) from exc

    if config == "default":
        dataset = datasets_module.load_dataset(dataset_id, split=split)
    else:
        dataset = datasets_module.load_dataset(dataset_id, config, split=split)
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return [dict(row) for row in dataset]


def load_hf_viewer_rows(
    dataset_id: str, split: str, limit: int, *, config: str = "default"
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    total: int | None = None
    target = limit if limit else None
    while total is None or offset < total:
        length = HF_VIEWER_ROWS_PAGE_SIZE
        if target is not None:
            remaining = target - len(rows)
            if remaining <= 0:
                break
            length = min(length, remaining)
        query = urllib.parse.urlencode(
            {
                "dataset": dataset_id,
                "config": config,
                "split": split,
                "offset": offset,
                "length": length,
            }
        )
        url = f"https://datasets-server.huggingface.co/rows?{query}"
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if payload.get("error"):
            raise RuntimeError(f"HF Dataset Viewer error for {dataset_id}: {payload}")
        page_rows = [dict(item["row"]) for item in payload.get("rows") or []]
        rows.extend(page_rows)
        total = int(payload.get("num_rows_total") or len(rows))
        if not page_rows:
            break
        offset += len(page_rows)
        if target is not None and len(rows) >= target:
            break
    return rows


def load_nisq_rows(limit: int) -> list[dict[str, Any]]:
    with urllib.request.urlopen(NISQ_DATASET_URL, timeout=60) as response:
        payload = response.read().decode("utf-8-sig")

    rows: list[dict[str, Any]] = []
    reader = csv.DictReader(payload.splitlines(), delimiter=";")
    for row in reader:
        question = row.get("question")
        label = row.get("label")
        if not question or not label:
            continue
        normalized_label = (
            "FACT_CHECK_NEEDED"
            if label.strip().upper() == "ISQ"
            else "NO_FACT_CHECK_NEEDED"
        )
        rows.append(
            {
                "id": f"nisq-{row.get('index') or len(rows)}",
                "text": question.strip(),
                "label": normalized_label,
                "source": "factcheck-nisq",
                "raw_label": label,
            }
        )
    return stratified_limit(rows, limit)


def stratified_limit(
    rows: list[dict[str, Any]], limit: int, *, label_field: str = "label"
) -> list[dict[str, Any]]:
    if not limit or len(rows) <= limit:
        return rows

    labels: list[str] = []
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        label = str(row.get(label_field))
        if label not in buckets:
            labels.append(label)
            buckets[label] = []
        buckets[label].append(row)

    selected: list[dict[str, Any]] = []
    offset = 0
    while len(selected) < limit:
        added = False
        for label in labels:
            bucket = buckets[label]
            if offset < len(bucket):
                selected.append(bucket[offset])
                added = True
                if len(selected) >= limit:
                    break
        if not added:
            break
        offset += 1
    return selected


def load_jailbreak_toxic_chat_rows(limit: int) -> list[dict[str, Any]]:
    rows = load_hf_config_rows(
        "lmsys/toxic-chat", "test", 0, config="toxicchat0124"
    )
    normalized = []
    for idx, row in enumerate(rows):
        label = "jailbreak" if int(row.get("jailbreaking") or 0) > 0 else "benign"
        normalized.append(
            {
                "id": row.get("conv_id") or f"toxic-chat-{idx}",
                "text": row.get("user_input") or "",
                "label": label,
                "source": "jailbreak-toxic-chat",
                "raw": row,
            }
        )
    return stratified_limit(normalized, limit)


def load_jailbreak_salad_rows(limit: int) -> list[dict[str, Any]]:
    target = limit or HF_VIEWER_ROWS_PAGE_SIZE
    rows = load_hf_config_rows(
        "OpenSafetyLab/Salad-Data",
        "train",
        target,
        config="attack_enhanced_set",
    )
    normalized = []
    for idx, row in enumerate(rows):
        normalized.append(
            {
                "id": f"salad-{row.get('aid') or idx}-{row.get('qid') or idx}",
                "text": row.get("augq") or row.get("baseq") or "",
                "label": "jailbreak",
                "source": "jailbreak-salad",
                "raw": row,
            }
        )
    return normalized


def load_jailbreak_mixed_rows(limit: int) -> list[dict[str, Any]]:
    target = limit or HF_VIEWER_ROWS_PAGE_SIZE
    negative_target = max(1, target // 2)
    positive_target = max(1, target - negative_target)
    toxic_rows = load_jailbreak_toxic_chat_rows(0)
    toxic_benign = [row for row in toxic_rows if row["label"] == "benign"]
    toxic_positive = [row for row in toxic_rows if row["label"] == "jailbreak"]
    salad_needed = max(0, positive_target - len(toxic_positive))
    salad_positive = load_jailbreak_salad_rows(salad_needed) if salad_needed else []
    rows = toxic_benign[:negative_target] + (
        toxic_positive + salad_positive
    )[:positive_target]
    return stratified_limit(rows, target)


def load_dataset_rows(spec: str, limit: int) -> list[dict[str, Any]]:
    e2e_datasets = {
        "e2e-domain": REPO_ROOT / "e2e/testcases/testdata/domain_classify_cases.json",
        "e2e-pii": REPO_ROOT / "e2e/testcases/testdata/pii_detection_cases.json",
        "e2e-jailbreak": REPO_ROOT
        / "e2e/testcases/testdata/jailbreak_detection_cases.json",
    }
    if spec in e2e_datasets:
        rows = read_json_or_jsonl(e2e_datasets[spec])
        return rows[:limit] if limit else rows

    builtin_loaders = {
        "factcheck-nisq": load_nisq_rows,
        "jailbreak-toxic-chat": load_jailbreak_toxic_chat_rows,
        "jailbreak-salad": load_jailbreak_salad_rows,
        "jailbreak-mixed": load_jailbreak_mixed_rows,
    }
    if spec in builtin_loaders:
        return builtin_loaders[spec](limit)
    if spec.startswith("json:"):
        rows = read_json_or_jsonl(Path(spec[5:]))
        return rows[:limit] if limit else rows
    if spec.startswith("jsonl:"):
        rows = read_json_or_jsonl(Path(spec[6:]))
        return rows[:limit] if limit else rows
    if spec.startswith("hf:"):
        return load_hf_rows(spec, limit)
    path = Path(spec)
    if path.exists():
        rows = read_json_or_jsonl(path)
        return rows[:limit] if limit else rows
    raise ValueError(f"Unsupported dataset spec: {spec}")


def first_present(row: dict[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return None


def normalize_examples(
    rows: list[dict[str, Any]],
    *,
    task_type: str,
    dataset_spec: str,
    text_field: str | None,
    label_field: str | None,
    entities_field: str,
) -> list[EvalExample]:
    examples: list[EvalExample] = []
    for idx, row in enumerate(rows):
        text = first_present(
            row,
            [text_field] if text_field else [],
        )
        if text is None:
            text = first_present(
                row,
                (
                    "text",
                    "source_text",
                    "question",
                    "prompt",
                    "user_input",
                    "input",
                ),
            )
        if text is None:
            raise ValueError(
                f"Dataset row {idx} has no text/question/prompt field: {row}"
            )

        label: str | int | None = first_present(
            row,
            [label_field] if label_field else [],
        )
        if label is None and task_type == "sequence":
            label = first_present(
                row,
                (
                    "label_name",
                    "label",
                    "gold_label",
                    "gold_intent",
                    "category",
                    "expected_label",
                ),
            )
        if label is None and "expected_blocked" in row:
            if "jailbreak" in dataset_spec:
                label = "jailbreak" if row["expected_blocked"] else "benign"
            else:
                label = "blocked" if row["expected_blocked"] else "allowed"

        entities = first_present(
            row, (entities_field, "privacy_mask", "spans", "pii_entities")
        )
        normalized_entities: tuple[dict[str, Any], ...] = ()
        if isinstance(entities, list):
            normalized_entities = tuple(normalize_entity(entity) for entity in entities)
        elif task_type == "token" and row.get("expected_blocked"):
            entity_type = first_present(row, ("pii_type", "entity_type", "type"))
            if entity_type:
                normalized_entities = (
                    {
                        "type": str(entity_type),
                        "start": None,
                        "end": None,
                        "text": None,
                    },
                )

        examples.append(
            EvalExample(
                id=str(first_present(row, ("id", "name")) or idx),
                text=str(text),
                label=label,
                entities=normalized_entities,
                source=str(first_present(row, ("source",)) or dataset_spec),
                raw=row,
            )
        )
    return examples


def normalize_entity(entity: dict[str, Any]) -> dict[str, Any]:
    entity_type = first_present(entity, ("type", "entity_type", "label", "pii_type"))
    return {
        "type": str(entity_type) if entity_type is not None else "",
        "start": first_present(entity, ("start", "start_char")),
        "end": first_present(entity, ("end", "end_char")),
        "text": first_present(entity, ("text", "value")),
    }


def load_tokenizer(model_dir: Path, tokenizer_dir: Path | None):
    auto_tokenizer, fast_tokenizer = require_transformers()
    source = tokenizer_dir or model_dir
    try:
        tokenizer = auto_tokenizer.from_pretrained(
            source, local_files_only=source.exists()
        )
    except Exception:
        tokenizer_json = source if source.is_file() else source / "tokenizer.json"
        if not tokenizer_json.exists():
            tokenizer_json = model_dir / "onnx/tokenizer.json"
        if not tokenizer_json.exists():
            raise
        tokenizer = fast_tokenizer(tokenizer_file=str(tokenizer_json))
    if tokenizer.pad_token is None and (tokenizer.eos_token or tokenizer.unk_token):
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    return tokenizer


def load_model_config(model_dir: Path) -> tuple[dict[int, str], dict[str, int], int]:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        config_path = model_dir / "onnx/config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found under {model_dir}")

    with config_path.open(encoding="utf-8") as f:
        config = json.load(f)

    raw_id2label = config.get("id2label") or {}
    raw_label2id = config.get("label2id") or {}
    id2label = {int(k): str(v) for k, v in raw_id2label.items()}
    label2id = {str(k): int(v) for k, v in raw_label2id.items()}
    if not id2label and label2id:
        id2label = {v: k for k, v in label2id.items()}
    if not label2id and id2label:
        label2id = {v: k for k, v in id2label.items()}
    if not id2label:
        raise ValueError(f"No id2label/label2id mapping in {config_path}")
    return id2label, label2id, int(config.get("pad_token_id", 0))


def parse_run_specs(values: list[str], model_dir: Path) -> list[RunSpec]:
    specs = []
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"Invalid --run {value!r}; expected NAME=ARTIFACT:PROVIDER[:RUNTIME]"
            )
        name, rest = value.split("=", 1)
        parts = rest.split(":")
        if len(parts) not in (
            RUN_SPEC_ARTIFACT_PROVIDER_PARTS,
            RUN_SPEC_RUNTIME_PARTS,
        ):
            raise ValueError(
                f"Invalid --run {value!r}; expected NAME=ARTIFACT:PROVIDER[:RUNTIME]"
            )
        artifact = resolve_artifact(model_dir, parts[0])
        provider = parts[1].lower()
        runtime = parts[2] if len(parts) == RUN_SPEC_RUNTIME_PARTS else "direct-ort"
        specs.append(
            RunSpec(name=name, artifact=artifact, provider=provider, runtime=runtime)
        )
    return specs


def resolve_artifact(model_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = [model_dir / value, model_dir / "onnx" / value]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def provider_chain(provider: str) -> tuple[list[str], str | None]:
    runtime = require_onnxruntime()
    available = set(runtime.get_available_providers())
    desired: list[str]
    if provider == "cpu":
        desired = ["CPUExecutionProvider"]
    elif provider == "migraphx":
        desired = ["MIGraphXExecutionProvider", "CPUExecutionProvider"]
    elif provider == "rocm":
        desired = ["ROCMExecutionProvider", "CPUExecutionProvider"]
    elif provider == "auto":
        desired = [
            "MIGraphXExecutionProvider",
            "ROCMExecutionProvider",
            "CPUExecutionProvider",
        ]
    else:
        raise ValueError(f"Unsupported provider {provider!r}")

    selected = [name for name in desired if name in available]
    if not selected:
        raise RuntimeError(
            f"None of requested providers {desired} are available; available={sorted(available)}"
        )
    fallback_reason = None
    if selected[0] != desired[0]:
        fallback_reason = f"{desired[0]} unavailable; selected {selected[0]}"
    return selected, fallback_reason


def create_session(spec: RunSpec) -> tuple[Any, list[str], str | None]:
    if not spec.artifact.exists():
        raise FileNotFoundError(f"ONNX artifact not found: {spec.artifact}")
    runtime = require_onnxruntime()
    providers, fallback_reason = provider_chain(spec.provider)
    session = runtime.InferenceSession(str(spec.artifact), providers=providers)
    selected = session.get_providers()
    if (
        providers
        and selected
        and selected[0] != providers[0]
        and fallback_reason is None
    ):
        fallback_reason = (
            f"ORT selected {selected[0]} instead of requested {providers[0]}"
        )
    return session, selected, fallback_reason


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float32)
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def extract_output_array(outputs: list[np.ndarray]) -> np.ndarray:
    if not outputs:
        raise RuntimeError("ONNX session returned no outputs")
    array = np.asarray(outputs[0])
    return array.astype(np.float32)


def run_sequence(
    spec: RunSpec,
    session: Any,
    tokenizer,
    examples: list[EvalExample],
    id2label: dict[int, str],
    *,
    batch_size: int,
    max_length: int,
    pad_token_id: int,
    pad_to_max_length: bool,
) -> tuple[list[dict[str, Any]], list[float]]:
    predictions: list[dict[str, Any]] = []
    latency_ms: list[float] = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        encoded = tokenizer(
            [example.text for example in batch],
            padding=False,
            truncation=True,
            max_length=max_length,
        )
        token_ids = encoded["input_ids"]
        masks = encoded["attention_mask"]
        max_len = max_length if pad_to_max_length else max(len(ids) for ids in token_ids)
        input_ids = np.full((len(batch), max_len), pad_token_id, dtype=np.int64)
        attention_mask = np.zeros((len(batch), max_len), dtype=np.int64)
        for row, (ids, mask) in enumerate(zip(token_ids, masks, strict=False)):
            seq_len = min(len(ids), max_len)
            input_ids[row, :seq_len] = np.asarray(ids, dtype=np.int64)
            attention_mask[row, :seq_len] = np.asarray(mask, dtype=np.int64)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        t0 = time.perf_counter()
        outputs = session.run(None, inputs)
        elapsed = (time.perf_counter() - t0) * 1000.0
        latency_ms.append(elapsed)

        logits = extract_output_array(outputs)
        if logits.ndim != SEQUENCE_LOGITS_RANK:
            logits = logits.reshape((len(batch), -1))
        probs = softmax(logits)
        for offset, example in enumerate(batch):
            pred_id = int(np.argmax(probs[offset]))
            predictions.append(
                {
                    "id": example.id,
                    "label": id2label.get(pred_id, str(pred_id)),
                    "label_id": pred_id,
                    "confidence": float(probs[offset, pred_id]),
                    "probs": probs[offset].astype(float).tolist(),
                    "logits": logits[offset].astype(float).tolist(),
                    "runtime": spec.runtime,
                }
            )
    return predictions, latency_ms


def run_token(
    spec: RunSpec,
    session: Any,
    tokenizer,
    examples: list[EvalExample],
    id2label: dict[int, str],
    *,
    max_length: int,
    pad_token_id: int,
    pad_to_max_length: bool,
) -> tuple[list[dict[str, Any]], list[float]]:
    predictions: list[dict[str, Any]] = []
    latency_ms: list[float] = []
    for example in examples:
        encoded = tokenizer(
            example.text,
            padding=False,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
            return_tensors="np",
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        input_ids = encoded["input_ids"].astype(np.int64)
        attention_mask = encoded["attention_mask"].astype(np.int64)
        if pad_to_max_length and input_ids.shape[1] < max_length:
            padded_input_ids = np.full((1, max_length), pad_token_id, dtype=np.int64)
            padded_attention_mask = np.zeros((1, max_length), dtype=np.int64)
            seq_len = input_ids.shape[1]
            padded_input_ids[:, :seq_len] = input_ids
            padded_attention_mask[:, :seq_len] = attention_mask
            input_ids = padded_input_ids
            attention_mask = padded_attention_mask
            offsets.extend([[0, 0]] * (max_length - len(offsets)))
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        t0 = time.perf_counter()
        outputs = session.run(None, inputs)
        elapsed = (time.perf_counter() - t0) * 1000.0
        latency_ms.append(elapsed)

        logits = extract_output_array(outputs)
        if logits.ndim == TOKEN_LOGITS_RANK:
            logits = logits[0]
        elif logits.ndim > TOKEN_LOGITS_RANK:
            logits = np.squeeze(logits)
            if logits.ndim == TOKEN_LOGITS_RANK:
                logits = logits[0]
        num_labels = len(id2label)
        if (
            logits.ndim == SEQUENCE_LOGITS_RANK
            and logits.shape[0] == 1
            and logits.shape[1] == num_labels
            and len(offsets) > 1
        ):
            raise RuntimeError(
                "Token task received sequence-classification logits "
                f"shape {logits.shape}; expected [seq_len, num_labels] or "
                "[batch, seq_len, num_labels]."
            )
        if logits.ndim != SEQUENCE_LOGITS_RANK:
            raise RuntimeError(
                f"Expected token logits rank 2/3, got shape {logits.shape}"
            )
        if logits.shape[0] < len(offsets):
            raise RuntimeError(
                f"Token logits length {logits.shape[0]} is shorter than "
                f"tokenizer offsets length {len(offsets)}; artifact is not "
                "compatible with token classification."
            )

        entities = bio_decode(example.text, offsets, logits, id2label)
        predictions.append(
            {
                "id": example.id,
                "entities": entities,
                "has_pii": bool(entities),
                "runtime": spec.runtime,
            }
        )
    return predictions, latency_ms


def bio_decode(
    text: str, offsets: list[list[int]], logits: np.ndarray, id2label: dict[int, str]
) -> list[dict[str, Any]]:
    probs = softmax(logits)
    entities: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    def flush() -> None:
        nonlocal current
        if not current:
            return
        start = current["start"]
        end = current["end"]
        if start is not None and end is not None and 0 <= start < end <= len(text):
            current["text"] = text[start:end]
        entities.append(current)
        current = None

    for idx, row in enumerate(probs):
        if idx >= len(offsets):
            break
        start, end = offsets[idx]
        if start == 0 and end == 0:
            continue
        label_id = int(np.argmax(row))
        confidence = float(row[label_id])
        label = id2label.get(label_id, str(label_id))
        if label.startswith("B-"):
            flush()
            current = {
                "type": label[2:],
                "start": int(start),
                "end": int(end),
                "confidence": confidence,
            }
        elif label.startswith("I-"):
            entity_type = label[2:]
            if current and current["type"] == entity_type:
                current["end"] = int(end)
                current["confidence"] = min(float(current["confidence"]), confidence)
            else:
                flush()
                current = {
                    "type": entity_type,
                    "start": int(start),
                    "end": int(end),
                    "confidence": confidence,
                }
        else:
            flush()
    flush()
    return entities


def run_eval(
    spec: RunSpec,
    *,
    task_type: str,
    tokenizer,
    examples: list[EvalExample],
    id2label: dict[int, str],
    pad_token_id: int,
    batch_size: int,
    max_length: int,
    pad_to_max_length: bool,
) -> RunResult:
    try:
        session, selected, fallback = create_session(spec)
        if task_type == "sequence":
            predictions, latency = run_sequence(
                spec,
                session,
                tokenizer,
                examples,
                id2label,
                batch_size=batch_size,
                max_length=max_length,
                pad_token_id=pad_token_id,
                pad_to_max_length=pad_to_max_length,
            )
        else:
            predictions, latency = run_token(
                spec,
                session,
                tokenizer,
                examples,
                id2label,
                max_length=max_length,
                pad_token_id=pad_token_id,
                pad_to_max_length=pad_to_max_length,
            )
        return RunResult(spec, "ok", selected, fallback, None, predictions, latency)
    except Exception as exc:
        return RunResult(spec, "error", [], None, str(exc), [], [])


def label_to_id_or_name(
    label: str | int | None, label2id: dict[str, int]
) -> str | None:
    if label is None:
        return None
    if isinstance(label, int):
        return str(label)
    label_str = str(label)
    if label_str in label2id:
        return label_str
    normalized = label_str.strip()
    if normalized in label2id:
        return normalized
    lowered = normalized.lower()
    for known in label2id:
        if known.lower() == lowered:
            return known
    return normalized


def classification_metrics(
    examples: list[EvalExample],
    predictions: list[dict[str, Any]],
    label2id: dict[str, int],
) -> dict[str, Any]:
    y_true: list[str] = []
    y_pred: list[str] = []
    for example, pred in zip(examples, predictions, strict=False):
        gold = label_to_id_or_name(example.label, label2id)
        if gold is None:
            continue
        y_true.append(gold)
        y_pred.append(
            label_to_id_or_name(pred["label"], label2id) or str(pred["label"])
        )
    if not y_true:
        return {"labeled_count": 0}

    labels = sorted(set(y_true) | set(y_pred))
    correct = sum(1 for a, b in zip(y_true, y_pred, strict=False) if a == b)
    per_label = {}
    f1_values = []
    recalls = {}
    for label in labels:
        tp = sum(
            1 for a, b in zip(y_true, y_pred, strict=False) if a == label and b == label
        )
        fp = sum(
            1 for a, b in zip(y_true, y_pred, strict=False) if a != label and b == label
        )
        fn = sum(
            1 for a, b in zip(y_true, y_pred, strict=False) if a == label and b != label
        )
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(1 for a in y_true if a == label),
        }
        f1_values.append(f1)
        recalls[label] = recall

    return {
        "labeled_count": len(y_true),
        "accuracy": correct / len(y_true),
        "macro_f1": statistics.fmean(f1_values) if f1_values else 0.0,
        "per_label": per_label,
        "recall": recalls,
        "confusion": {
            f"{a}->{b}": count
            for (a, b), count in Counter(zip(y_true, y_pred, strict=False)).items()
        },
    }


def entity_key(entity: dict[str, Any], exact_span: bool) -> tuple[Any, ...]:
    entity_type = str(entity.get("type") or entity.get("entity_type") or "")
    if exact_span and entity.get("start") is not None and entity.get("end") is not None:
        return (entity_type, int(entity["start"]), int(entity["end"]))
    text = entity.get("text")
    return (entity_type, str(text) if text is not None else None)


def token_metrics(
    examples: list[EvalExample], predictions: list[dict[str, Any]]
) -> dict[str, Any]:
    exact_tp = exact_fp = exact_fn = 0
    type_tp = type_fp = type_fn = 0
    any_correct = 0
    typed_support = 0
    typed_found = 0

    for example, pred in zip(examples, predictions, strict=False):
        gold_entities = list(example.entities)
        pred_entities = list(pred.get("entities") or [])
        gold_has = bool(gold_entities) or bool(example.raw.get("expected_blocked"))
        pred_has = bool(pred_entities)
        any_correct += int(gold_has == pred_has)

        gold_exact = {entity_key(entity, exact_span=True) for entity in gold_entities}
        pred_exact = {entity_key(entity, exact_span=True) for entity in pred_entities}
        exact_tp += len(gold_exact & pred_exact)
        exact_fp += len(pred_exact - gold_exact)
        exact_fn += len(gold_exact - pred_exact)

        gold_types = {
            str(entity.get("type") or "")
            for entity in gold_entities
            if entity.get("type")
        }
        pred_types = {
            str(entity.get("type") or "")
            for entity in pred_entities
            if entity.get("type")
        }
        type_tp += len(gold_types & pred_types)
        type_fp += len(pred_types - gold_types)
        type_fn += len(gold_types - pred_types)
        typed_support += len(gold_types)
        typed_found += len(gold_types & pred_types)

    def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        return {"precision": precision, "recall": recall, "f1": f1}

    return {
        "labeled_count": len(examples),
        "any_entity_accuracy": any_correct / len(examples) if examples else 0.0,
        "exact_entity": prf(exact_tp, exact_fp, exact_fn),
        "entity_type": prf(type_tp, type_fp, type_fn),
        "typed_gold_found": typed_found / typed_support if typed_support else None,
    }


def compare_to_baseline(
    baseline: RunResult, run: RunResult, *, task_type: str
) -> dict[str, Any]:
    if baseline.status != "ok" or run.status != "ok":
        return {"status": "skipped"}
    total = min(len(baseline.predictions), len(run.predictions))
    if total == 0:
        return {"status": "skipped"}

    if task_type == "sequence":
        matches = 0
        max_abs_values = []
        mean_abs_values = []
        mismatches = []
        for base_pred, pred in zip(baseline.predictions, run.predictions, strict=False):
            same = base_pred.get("label") == pred.get("label")
            matches += int(same)
            if not same and len(mismatches) < MISMATCH_SAMPLE_LIMIT:
                mismatches.append(
                    {
                        "id": pred.get("id"),
                        "baseline": base_pred.get("label"),
                        "candidate": pred.get("label"),
                        "baseline_confidence": base_pred.get("confidence"),
                        "candidate_confidence": pred.get("confidence"),
                    }
                )
            base_logits = np.asarray(base_pred.get("logits") or [], dtype=np.float32)
            logits = np.asarray(pred.get("logits") or [], dtype=np.float32)
            if base_logits.shape == logits.shape and base_logits.size:
                diff = np.abs(base_logits - logits)
                max_abs_values.append(float(np.max(diff)))
                mean_abs_values.append(float(np.mean(diff)))
        return {
            "status": "ok",
            "label_match_rate": matches / total,
            "mismatch_count": total - matches,
            "mismatches": mismatches,
            "logit_max_abs": max(max_abs_values) if max_abs_values else None,
            "logit_mean_abs": (
                statistics.fmean(mean_abs_values) if mean_abs_values else None
            ),
        }

    matches = 0
    mismatches = []
    for base_pred, pred in zip(baseline.predictions, run.predictions, strict=False):
        base_entities = {
            entity_key(entity, exact_span=True)
            for entity in base_pred.get("entities", [])
        }
        entities = {
            entity_key(entity, exact_span=True) for entity in pred.get("entities", [])
        }
        same = base_entities == entities
        matches += int(same)
        if not same and len(mismatches) < MISMATCH_SAMPLE_LIMIT:
            mismatches.append(
                {
                    "id": pred.get("id"),
                    "baseline_entities": sorted(map(str, base_entities)),
                    "candidate_entities": sorted(map(str, entities)),
                }
            )
    return {
        "status": "ok",
        "entity_match_rate": matches / total,
        "mismatch_count": total - matches,
        "mismatches": mismatches,
    }


def latency_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"avg_ms": None, "p50_ms": None, "p95_ms": None, "p99_ms": None}
    sorted_values = sorted(values)

    def percentile(p: float) -> float:
        if len(sorted_values) == 1:
            return sorted_values[0]
        pos = (len(sorted_values) - 1) * p
        low = math.floor(pos)
        high = math.ceil(pos)
        if low == high:
            return sorted_values[low]
        return sorted_values[low] * (high - pos) + sorted_values[high] * (pos - low)

    return {
        "avg_ms": statistics.fmean(values),
        "p50_ms": percentile(0.50),
        "p95_ms": percentile(0.95),
        "p99_ms": percentile(0.99),
    }


def write_jsonl(path: Path, events: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


def write_markdown(
    path: Path,
    *,
    task: str,
    task_type: str,
    dataset: str,
    model_dir: Path,
    baseline_name: str,
    run_summaries: list[dict[str, Any]],
    comparisons: dict[str, dict[str, Any]],
    regression_failures: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Router Signal Artifact Eval: {task}",
        "",
        f"- Task type: `{task_type}`",
        f"- Dataset: `{dataset}`",
        f"- Model dir: `{model_dir}`",
        f"- Baseline run: `{baseline_name}`",
        "",
        "## Runs",
        "",
        "| Run | Status | Runtime | Provider | Selected providers | Fallback | Accuracy | Macro F1 | Entity F1 | Label/entity match |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for summary in run_summaries:
        metrics = summary.get("metrics") or {}
        compare = comparisons.get(summary["name"], {})
        accuracy = fmt_metric(
            metrics.get("accuracy") or metrics.get("any_entity_accuracy")
        )
        macro_f1 = fmt_metric(metrics.get("macro_f1"))
        entity_f1 = fmt_metric((metrics.get("exact_entity") or {}).get("f1"))
        match = fmt_metric(
            compare.get("label_match_rate") or compare.get("entity_match_rate")
        )
        lines.append(
            "| {name} | {status} | `{runtime}` | `{provider}` | `{selected}` | {fallback} | {accuracy} | {macro_f1} | {entity_f1} | {match} |".format(
                name=summary["name"],
                status=summary["status"],
                runtime=summary["runtime"],
                provider=summary["provider"],
                selected=", ".join(summary.get("selected_providers") or []),
                fallback=summary.get("fallback_reason") or "",
                accuracy=accuracy,
                macro_f1=macro_f1,
                entity_f1=entity_f1,
                match=match,
            )
        )
    lines.extend(["", "## Baseline Comparisons", ""])
    for run_name, comparison in comparisons.items():
        if run_name == baseline_name:
            continue
        lines.append(f"### {run_name}")
        lines.append("")
        lines.append("```json")
        lines.append(
            json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True)
        )
        lines.append("```")
        lines.append("")

    if regression_failures:
        lines.extend(["## Regression Gate Failures", ""])
        for failure in regression_failures:
            lines.append(f"- {failure}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def fmt_metric(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return str(value)


def evaluate_regressions(
    *,
    baseline_summary: dict[str, Any],
    run_summaries: list[dict[str, Any]],
    comparisons: dict[str, dict[str, Any]],
    task_type: str,
    positive_labels: list[str],
    min_label_match: float,
    max_accuracy_drop: float,
    max_macro_f1_drop: float,
) -> list[str]:
    failures = []
    baseline_metrics = baseline_summary.get("metrics") or {}
    for summary in run_summaries:
        if summary["status"] != "ok":
            failures.append(
                f"{summary['name']}: run failed with error: {summary.get('error')}"
            )
            continue
        if summary["name"] == baseline_summary["name"]:
            continue
        metrics = summary.get("metrics") or {}
        comparison = comparisons.get(summary["name"], {})
        if task_type == "sequence":
            match = comparison.get("label_match_rate")
            if match is not None and match < min_label_match:
                failures.append(
                    f"{summary['name']}: label match {match:.6f} < {min_label_match:.6f}"
                )
            base_acc = baseline_metrics.get("accuracy")
            acc = metrics.get("accuracy")
            if (
                base_acc is not None
                and acc is not None
                and base_acc - acc > max_accuracy_drop
            ):
                failures.append(
                    f"{summary['name']}: accuracy drop {base_acc - acc:.6f} > {max_accuracy_drop:.6f}"
                )
            base_f1 = baseline_metrics.get("macro_f1")
            f1 = metrics.get("macro_f1")
            if (
                base_f1 is not None
                and f1 is not None
                and base_f1 - f1 > max_macro_f1_drop
            ):
                failures.append(
                    f"{summary['name']}: macro-F1 drop {base_f1 - f1:.6f} > {max_macro_f1_drop:.6f}"
                )
            for label in positive_labels:
                base_recall = (baseline_metrics.get("recall") or {}).get(label)
                recall = (metrics.get("recall") or {}).get(label)
                if (
                    base_recall is not None
                    and recall is not None
                    and recall < base_recall
                ):
                    failures.append(
                        f"{summary['name']}: positive recall for {label} dropped from {base_recall:.6f} to {recall:.6f}"
                    )
        else:
            match = comparison.get("entity_match_rate")
            if match is not None and match < min_label_match:
                failures.append(
                    f"{summary['name']}: entity match {match:.6f} < {min_label_match:.6f}"
                )
            base_recall = (baseline_metrics.get("exact_entity") or {}).get("recall")
            recall = (metrics.get("exact_entity") or {}).get("recall")
            if base_recall is not None and recall is not None and recall < base_recall:
                failures.append(
                    f"{summary['name']}: exact entity recall dropped from {base_recall:.6f} to {recall:.6f}"
                )
    return failures


def main() -> int:
    args = parse_args()
    runtime = require_onnxruntime()
    max_length = min(args.max_length, MAX_CLASSIFICATION_SEQ_LEN)
    rows = load_dataset_rows(args.dataset, args.limit)
    examples = normalize_examples(
        rows,
        task_type=args.task_type,
        dataset_spec=args.dataset,
        text_field=args.text_field,
        label_field=args.label_field,
        entities_field=args.entities_field,
    )
    id2label, label2id, pad_token_id = load_model_config(args.model_dir)
    tokenizer = load_tokenizer(args.model_dir, args.tokenizer_dir)
    run_specs = parse_run_specs(args.run, args.model_dir)
    baseline_name = args.baseline_run or run_specs[0].name

    events: list[dict[str, Any]] = [
        {
            "event": "inventory",
            "task": args.task,
            "task_type": args.task_type,
            "dataset": args.dataset,
            "model_dir": str(args.model_dir),
            "available_providers": runtime.get_available_providers(),
            "example_count": len(examples),
            "labels": id2label,
        }
    ]

    results = [
        run_eval(
            spec,
            task_type=args.task_type,
            tokenizer=tokenizer,
            examples=examples,
            id2label=id2label,
            pad_token_id=pad_token_id,
            batch_size=args.batch_size,
            max_length=max_length,
            pad_to_max_length=args.pad_to_max_length,
        )
        for spec in run_specs
    ]

    baseline = next(
        (result for result in results if result.spec.name == baseline_name), None
    )
    if baseline is None:
        raise SystemExit(f"Baseline run {baseline_name!r} was not defined")

    run_summaries: list[dict[str, Any]] = []
    comparisons: dict[str, dict[str, Any]] = {}

    for result in results:
        if args.task_type == "sequence" and result.status == "ok":
            metrics = classification_metrics(examples, result.predictions, label2id)
        elif result.status == "ok":
            metrics = token_metrics(examples, result.predictions)
        else:
            metrics = {}

        summary = {
            "event": "run_summary",
            "name": result.spec.name,
            "runtime": result.spec.runtime,
            "artifact": str(result.spec.artifact),
            "provider": result.spec.provider,
            "selected_providers": result.selected_providers,
            "fallback_reason": result.fallback_reason,
            "status": result.status,
            "error": result.error,
            "latency": latency_summary(result.latency_ms),
            "metrics": metrics,
        }
        run_summaries.append(summary)
        events.append(summary)

        comparison = compare_to_baseline(baseline, result, task_type=args.task_type)
        comparisons[result.spec.name] = comparison
        events.append(
            {
                "event": "baseline_comparison",
                "baseline": baseline.spec.name,
                "candidate": result.spec.name,
                **comparison,
            }
        )

        for example, pred in zip(examples, result.predictions, strict=False):
            events.append(
                {
                    "event": "prediction",
                    "run": result.spec.name,
                    "task": args.task,
                    "id": example.id,
                    "source": example.source,
                    "gold_label": example.label,
                    "gold_entities": list(example.entities),
                    "prediction": pred,
                }
            )

    baseline_summary = next(
        summary for summary in run_summaries if summary["name"] == baseline_name
    )
    regression_failures = evaluate_regressions(
        baseline_summary=baseline_summary,
        run_summaries=run_summaries,
        comparisons=comparisons,
        task_type=args.task_type,
        positive_labels=args.positive_label,
        min_label_match=args.min_label_match,
        max_accuracy_drop=args.max_accuracy_drop,
        max_macro_f1_drop=args.max_macro_f1_drop,
    )
    events.append({"event": "regression_gates", "failures": regression_failures})

    if args.jsonl:
        write_jsonl(args.jsonl, events)
    if args.summary_md:
        write_markdown(
            args.summary_md,
            task=args.task,
            task_type=args.task_type,
            dataset=args.dataset,
            model_dir=args.model_dir,
            baseline_name=baseline_name,
            run_summaries=run_summaries,
            comparisons=comparisons,
            regression_failures=regression_failures,
        )
    else:
        print(json.dumps(run_summaries, ensure_ascii=False, indent=2, sort_keys=True))

    if regression_failures:
        for failure in regression_failures:
            print(f"REGRESSION: {failure}", file=sys.stderr)
        if args.fail_on_regression:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
