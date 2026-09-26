"""Bounded, label-isolated long-context Qwen3 reranker pilot.

The published reranker's yes/no terminal logits score each candidate in the
same complete option set. Neither the prompt nor the collector sees gold.
Private TRAIN/SELECT rows are audited on the GPU host; this source contains no
row text or benchmark labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from training.model.data import check_partition_isolation, file_sha256, load_partition

MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
MODEL_REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
MODEL_FILES = {
    "config.json": "d479c427a9ca5295218063d4f9aca4f297ab4ac27487cca7af42c84643d51ef0",
    "model.safetensors": "27cd75a405b9c1b46b59abfd88aaa209e6fed2a1972cde9b70e7659537c5e65b",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
}
TRAIN_SHA = "61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755"
SELECT_SHA = "32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6"
SEED = "decision2-qwen3-rerank06-cleanv2-20260927-v1"
ADAPTER = "qwen3-rerank06-full-slate-v1"
MAX_TOKENS = 4096
QUOTAS = {
    ("choice", "structured"): 96,
    ("choice", "other"): 96,
    ("noul", "structured"): 96,
    ("noul", "other"): 96,
    ("score", "structured"): 96,
    ("score", "other"): 32,
}
LONG_MINIMUM = {"choice": 32, "noul": 32, "score": 8}
PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements "
    "based on the Query and the Instruct provided. Note that the answer can "
    'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
INSTRUCT = (
    "Judge whether the nominated answer is the single correct decision for "
    "this task, considering all listed alternatives, stated constraints, "
    "exceptions, and evidence."
)


def verify_source(path: Path) -> dict[str, Any]:
    for name, expected in MODEL_FILES.items():
        if not (path / name).is_file() or file_sha256(path / name) != expected:
            raise ValueError(f"Pinned reranker source mismatch: {name}")
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if (
        config.get("architectures") != ["Qwen3ForCausalLM"]
        or config.get("max_position_embeddings", 0) < MAX_TOKENS
    ):
        raise ValueError("Unexpected reranker architecture or context")
    return {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "files_sha256": MODEL_FILES,
        "declared_max_positions": config["max_position_embeddings"],
    }


def _text(value: Any) -> str:
    return (
        value
        if isinstance(value, str)
        else json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    )


def row_question(row: dict[str, Any]) -> dict[str, Any]:
    kind = row["task_type"]
    options = row["options"]
    if kind == "choice":
        criteria = {x["key"]: x["description"] for x in options}
    elif kind == "noul":
        criteria = None
    elif kind == "score":
        ordered = sorted(options, key=lambda x: int(x["key"]))
        if [x["key"] for x in ordered] != [str(i) for i in range(len(ordered))]:
            raise ValueError("Score levels are not contiguous")
        criteria = [x["description"] for x in ordered]
    else:
        raise ValueError(f"Unsupported type: {kind}")
    result = {"type": kind, "instructions": _text(row["instructions"])}
    if criteria is not None:
        result["criteria"] = criteria
    return result


def option_items(question: dict[str, Any]) -> list[tuple[str, str]]:
    kind = question["type"]
    if kind == "choice":
        items = [(str(k), _text(v)) for k, v in question["criteria"].items()]
    elif kind == "noul":
        items = [
            ("no", "The decision instruction is false"),
            ("yes", "The decision instruction is true"),
        ]
    elif kind == "score":
        items = [(str(i), _text(v)) for i, v in enumerate(question["criteria"])]
    else:
        raise ValueError(f"Unsupported type: {kind}")
    if len(items) < 2 or len({key for key, _ in items}) != len(items):
        raise ValueError("Candidate options must be distinct")
    return items


def candidate_texts(
    state: Any, question: dict[str, Any]
) -> tuple[list[str], list[str]]:
    options = option_items(question)
    all_options = json.dumps(options, ensure_ascii=False, separators=(",", ":"))
    head = (
        f"Decision type: {question['type']}\n"
        f"Decision instructions: {question['instructions']}\n"
        f"All candidate answers: {all_options}\n"
    )
    doc = _text(state)
    return [
        PREFIX
        + f"<Instruct>: {INSTRUCT}\n<Query>: {head}Nominated answer: {key}\n<Document>: {doc}"
        + SUFFIX
        for key, _ in options
    ], [key for key, _ in options]


def encode(
    tokenizer: Any, state: Any, question: dict[str, Any], max_tokens: int = MAX_TOKENS
) -> tuple[list[list[int]], list[str]]:
    prompts, keys = candidate_texts(state, question)
    sequences = [
        tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts
    ]
    if any(len(ids) > max_tokens for ids in sequences):
        raise ValueError("context_overflow")
    return sequences, keys


def _rank(row: dict[str, Any]) -> str:
    return hashlib.sha256(f"{SEED}:{row['id']}".encode()).hexdigest()


def choose_rows(eligible: list[tuple[dict[str, Any], int]]) -> list[dict[str, Any]]:
    pools: dict[tuple[str, str], list[tuple[dict[str, Any], int]]] = defaultdict(list)
    for row, length in eligible:
        source_class = "structured" if row["family"].startswith("stage4_") else "other"
        pools[row["task_type"], source_class].append((row, length))
    selected = []
    for (kind, source_class), quota in QUOTAS.items():
        pool = pools[kind, source_class]
        long_minimum = LONG_MINIMUM[kind] if source_class == "structured" else 0
        long_pool = sorted(
            (item for item in pool if item[1] > 512), key=lambda item: _rank(item[0])
        )
        if len(long_pool) < long_minimum:
            raise ValueError(f"Insufficient long structured {kind} rows")
        first = long_pool[:long_minimum]
        seen = {row["id"] for row, _ in first}
        rest = sorted(
            (item for item in pool if item[0]["id"] not in seen),
            key=lambda item: _rank(item[0]),
        )
        chosen = first + rest[: quota - long_minimum]
        if len(chosen) != quota:
            raise ValueError(f"Insufficient {kind}/{source_class} rows")
        selected.extend(row for row, _ in chosen)
    return sorted(selected, key=lambda row: row["id"])


def target_key(row: dict[str, Any]) -> str:
    key = row["options"][row["label"]]["key"]
    if row["task_type"] == "noul":
        return {"false": "no", "true": "yes"}[key]
    return key


def build(source: Path, train: Path, select: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    source_identity = verify_source(source)
    if file_sha256(train) != TRAIN_SHA or file_sha256(select) != SELECT_SHA:
        raise ValueError("Frozen rights-clean data changed")
    train_rows = load_partition(train, "train")
    select_rows = load_partition(select, "select")
    check_partition_isolation({"train": train_rows, "select": select_rows})
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(source)
    eligible = []
    excluded = Counter()
    for row in train_rows:
        try:
            sequences, _ = encode(tokenizer, row["state"], row_question(row))
            eligible.append((row, max(map(len, sequences))))
        except ValueError as exc:
            if str(exc) != "context_overflow":
                raise
            excluded[row["task_type"]] += 1
    chosen = choose_rows(eligible)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for row in chosen:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    lengths = {}
    for row in chosen:
        sequences, keys = encode(tokenizer, row["state"], row_question(row))
        if target_key(row) not in keys:
            raise ValueError("Gold target absent from candidate set")
        lengths[row["id"]] = max(map(len, sequences))
    return {
        "contract": ADAPTER,
        "source": source_identity,
        "train_sha256": TRAIN_SHA,
        "select_sha256": SELECT_SHA,
        "output_sha256": file_sha256(output),
        "rows": len(chosen),
        "by_type": dict(Counter(row["task_type"] for row in chosen)),
        "by_source_class": dict(
            Counter(
                "structured" if row["family"].startswith("stage4_") else "other"
                for row in chosen
            )
        ),
        "long_over_512": sum(length > 512 for length in lengths.values()),
        "max_tokens": max(lengths.values()),
        "excluded_over_4096": dict(excluded),
    }


def project(
    question: dict[str, Any], keys: list[str], logits: list[float]
) -> dict[str, Any]:
    if len(keys) != len(logits) or any(not math.isfinite(x) for x in logits):
        raise ValueError("Malformed reranker scores")
    shift = max(logits)
    weights = [math.exp(x - shift) for x in logits]
    normalizer = sum(weights)
    probabilities = [x / normalizer for x in weights]
    best = max(range(len(keys)), key=probabilities.__getitem__)
    kind = question["type"]
    if kind == "choice":
        return {
            "type": kind,
            "choice": keys[best],
            "probabilities": dict(zip(keys, probabilities)),
            "confidence": probabilities[best],
        }
    if kind == "noul":
        return {
            "type": kind,
            "noul": probabilities[keys.index("yes")],
            "confidence": probabilities[best],
        }
    return {
        "type": kind,
        "native_level": best,
        "score": sum(i * p for i, p in enumerate(probabilities)),
        "probabilities": dict(zip(keys, probabilities)),
        "confidence": probabilities[best],
    }


def score_select(rows: list[dict[str, Any]], predictions: Path) -> dict[str, Any]:
    with predictions.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    if len(records) != len(rows) or [r["id"] for r in records] != [
        r["id"] for r in rows
    ]:
        raise ValueError("SELECT prediction IDs/order differ")
    by_type: dict[str, list[int]] = defaultdict(list)
    by_family: dict[str, list[int]] = defaultdict(list)
    invalid = 0
    for row, record in zip(rows, records):
        answer = record["answer"]
        kind = row["task_type"]
        key = target_key(row)
        if answer.get("type") != kind:
            raise ValueError("Wrong answer type")
        if "error" in answer:
            invalid += 1
            correct = 0
        elif kind == "choice":
            correct = int(answer.get("choice") == key)
        elif kind == "noul":
            p = answer.get("noul")
            correct = int(
                isinstance(p, (int, float))
                and p != 0.5
                and ("yes" if p > 0.5 else "no") == key
            )
        else:
            correct = int(answer.get("native_level") == int(key))
        by_type[kind].append(correct)
        by_family[row["family"]].append(correct)
    family = {
        k: {"correct": sum(v), "total": len(v)} for k, v in sorted(by_family.items())
    }
    return {
        "prediction_sha256": file_sha256(predictions),
        "correct": sum(sum(v) for v in by_type.values()),
        "total": len(rows),
        "invalid": invalid,
        "by_type": {
            k: {"correct": sum(v), "total": len(v)} for k, v in sorted(by_type.items())
        },
        "by_family": family,
        "family_macro_accuracy": sum(v["correct"] / v["total"] for v in family.values())
        / len(family),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_subparsers(dest="command", required=True)
    build_cmd = command.add_parser("build")
    for name in ("source", "train", "select", "output", "manifest"):
        build_cmd.add_argument(f"--{name}", type=Path, required=True)
    score_cmd = command.add_parser("score-select")
    score_cmd.add_argument("--select", type=Path, required=True)
    score_cmd.add_argument("--predictions", type=Path, required=True)
    score_cmd.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        result = build(args.source, args.train, args.select, args.output)
        args.manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        if file_sha256(args.select) != SELECT_SHA:
            raise ValueError("SELECT changed")
        result = score_select(load_partition(args.select, "select"), args.predictions)
        args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
