"""Gold-free zero-step native parity for the 9B Score-cardinality arm.

The fixed control is its archived SELECT baseline, never benchmark answers.
The emitted receipt contains counts and digests but no row text or identifiers.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import torch
from training.model.data import (
    check_partition_isolation,
    file_sha256,
    load_partition,
)
from training.model.decision_model import (
    PROMPT_VERSION,
    DecisionModel,
    collate,
    encode,
)
from training.model.lora import attach_lora
from training.model.source import source_fingerprint

SOURCE_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
TRAIN_SHA = "fe9c419a3e751e4a4173b5c90c49e0a83bba7e1c35c683441bb036138acb971c"
SELECT_SHA = "32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6"
CAL_SHA = "3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a"
BASELINE_SHA = "f2838de7611c8d3296825f81ff41d839d3a1573863801e02e5ce400b0f800133"
SOURCE_CONFIG_SHA = "d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05"
SOURCE_TOKENIZER_SHA = (
    "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
)
SOURCE_SHARDS = (
    "db6f444b43d318c92f360a13a25561a6a65b10c0631b8ed305a426dbaa6c380e",
    "31c7d7e2dd5d207840b31cc59083c8f4c4718959149e0358c0364052bb9a0330",
    "7ec36ba3a4176a44c3c0876ad80c56a2f70c84bf008d82e9501df642f17dadec",
    "b62b0c4cd7e44edee103ee8f4fe225f246d5e768e07bfd5f25b63a8aa1fdd0c6",
)


def _baseline_probs(record: dict, keys: list[str]) -> list[float]:
    if record["task_type"] == "noul":
        true = float(record["answer"]["noul"])
        return [true if key == "true" else 1.0 - true for key in keys]
    return [float(record["answer"]["probabilities"][key]) for key in keys]


def _predict(model: DecisionModel, items: list[dict], pad_id: int) -> list[list[float]]:
    out: list[list[float]] = []
    with torch.inference_mode():
        for start in range(0, len(items), 2):
            subset = items[start : start + 2]
            batch = {
                key: value.to("cuda:0") if torch.is_tensor(value) else value
                for key, value in collate(subset, pad_id).items()
            }
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(**batch)
            probs = logits.float().softmax(-1).cpu().tolist()
            out.extend(row[: len(item["keys"])] for row, item in zip(probs, subset))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--select", type=Path, required=True)
    parser.add_argument("--cal", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    for path, expected in (
        (args.train, TRAIN_SHA),
        (args.select, SELECT_SHA),
        (args.cal, CAL_SHA),
        (args.baseline, BASELINE_SHA),
        (args.source / "config.json", SOURCE_CONFIG_SHA),
        (args.source / "tokenizer.json", SOURCE_TOKENIZER_SHA),
    ):
        if file_sha256(path) != expected:
            raise ValueError("Pinned source or partition hash differs")
    fingerprint = source_fingerprint(args.source)
    shard_hashes = sorted(
        value
        for name, value in fingerprint["files_sha256"].items()
        if name.startswith("model-") and name.endswith(".safetensors")
    )
    if shard_hashes != sorted(SOURCE_SHARDS):
        raise ValueError("Official source weight shards differ")
    partitions = {
        name: load_partition(path, name)
        for name, path in (
            ("train", args.train),
            ("select", args.select),
            ("cal", args.cal),
        )
    }
    check_partition_isolation(partitions)
    train, select = partitions["train"], partitions["select"]
    score3 = [
        row for row in train if row["task_type"] == "score" and len(row["options"]) == 3
    ]
    score3_labels = Counter(row["options"][row["label"]]["key"] for row in score3)
    if (
        (len(train), len(select), len(partitions["cal"])) != (7324, 700, 700)
        or len({row["group_id"] for row in train}) != 5261
        or len(score3) != 99
        or len({row["group_id"] for row in score3}) != 99
        or score3_labels != {"0": 30, "1": 33, "2": 36}
    ):
        raise ValueError("CPU Score-cardinality roster differs")
    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(args.source, local_files_only=True)
    if config.model_type != "qwen3_5":
        raise ValueError("Official source architecture differs")
    cpu_tokenizer = AutoTokenizer.from_pretrained(args.source, local_files_only=True)
    lengths = [len(encode(row, cpu_tokenizer, 4096)["ids"]) for row in train]
    if sum(lengths) != 3_579_176 or max(lengths) != 4089:
        raise ValueError("Exact 9B native token exposure differs")
    selected_score3 = sorted(score3, key=lambda row: row["input_sha256"])[:3]
    baseline = [json.loads(line) for line in args.baseline.open(encoding="utf-8")]
    if len(baseline) != 700:
        raise ValueError("Archived SELECT baseline count differs")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A single BF16 ROCm/CUDA device is required")
    random.seed(20260926)
    torch.manual_seed(20260926)
    torch.cuda.manual_seed_all(20260926)
    model, tokenizer = DecisionModel.from_base(
        args.source,
        SOURCE_REVISION,
        256,
        source_stage="posttrained",
        head_variant="score-cardinality",
    )
    attach_lora(
        model,
        rank=16,
        alpha=32,
        dropout=0.05,
        source_kind="posttrained",
        source_fingerprint=fingerprint,
    )
    model = model.float().to("cuda:0").eval()
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    if pad_id is None:
        raise ValueError("Tokenizer has no pad/EOS token")
    selected = [encode(row, tokenizer, 4096) for row in select[:32]]
    score_items = [encode(row, tokenizer, 4096) for row in selected_score3]
    probabilities = _predict(model, selected, pad_id)
    drift = 0.0
    changed = 0
    for record, item, actual in zip(baseline[:32], selected, probabilities):
        if (
            record["id"] != item["id"]
            or record["prompt_sha256"] != item["prompt_sha256"]
            or record["token_ids_sha256"] != item["token_ids_sha256"]
        ):
            raise ValueError("Archived baseline input identity differs")
        old = _baseline_probs(record, item["keys"])
        drift = max(drift, *(abs(a - b) for a, b in zip(actual, old)))
        changed += actual.index(max(actual)) != old.index(max(old))
    residual = _predict(model, score_items, pad_id)
    head, variant = model.head, model.metadata["head_variant"]
    model.head, model.metadata["head_variant"] = head.shared, "shared"
    try:
        shared = _predict(model, score_items, pad_id)
    finally:
        model.head, model.metadata["head_variant"] = head, variant
    score_drift = max(
        abs(a - b)
        for current, control in zip(residual, shared)
        for a, b in zip(current, control)
    )
    score_changed = sum(
        current.index(max(current)) != control.index(max(control))
        for current, control in zip(residual, shared)
    )
    if changed or score_changed or drift > 1e-6 or score_drift > 1e-6:
        raise ValueError("Zero-step native parity gate failed")
    result = {
        "version": "qwen35-9b-score-cardinality-zero-step/1",
        "prompt_version": PROMPT_VERSION,
        "source_revision": SOURCE_REVISION,
        "data_sha256": {"train": TRAIN_SHA, "select": SELECT_SHA, "cal": CAL_SHA},
        "archived_baseline_sha256": BASELINE_SHA,
        "train_rows": len(train),
        "score3_rows": len(score3),
        "score3_groups": len({row["group_id"] for row in score3}),
        "native_train_tokens": sum(lengths),
        "native_train_max_length": max(lengths),
        "select_checked": len(selected),
        "score3_checked": len(score_items),
        "select_changed": changed,
        "score3_changed": score_changed,
        "select_max_probability_drift": drift,
        "score3_max_probability_drift": score_drift,
        "gpu_seconds": time.perf_counter() - started,
        "status": "PASS",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
