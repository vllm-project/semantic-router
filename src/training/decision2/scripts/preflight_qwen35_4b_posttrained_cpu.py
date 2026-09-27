"""Compare pinned official Qwen3.5 4B Base/Posttrained native input contracts.

This CPU-only audit never loads model weights, predicts, scores, or reads
JevArena/CSS evaluation labels. It is a prerequisite, not training admission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from math import prod
from pathlib import Path
from typing import Any

BASE_REVISION = "1001bb4d826a52d1f399e183466143f4da7b741b"
POSTTRAINED_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
DATA = {
    "train": (7455, "61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755"),
    "select": (700, "32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6"),
    "cal": (700, "3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a"),
}
MAX_LENGTH = 8192


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def load_rows(path: Path, role: str) -> list[dict[str, Any]]:
    count, expected = DATA[role]
    if sha256_file(path) != expected:
        raise ValueError(f"{role} data digest changed")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != count or len({row["id"] for row in rows}) != count:
        raise ValueError(f"{role} row count or ID uniqueness changed")
    return rows


def compare_metadata(base: Path, posttrained: Path) -> dict[str, Any]:
    base_config = json.loads((base / "config.json").read_text(encoding="utf-8"))
    post_config = json.loads((posttrained / "config.json").read_text(encoding="utf-8"))
    if base_config != post_config:
        raise ValueError("Base and posttrained model configs differ")
    if base_config.get("architectures") != ["Qwen3_5ForConditionalGeneration"]:
        raise ValueError("Expected official Qwen3.5 conditional architecture")
    base_index = json.loads(
        (base / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    post_index = json.loads(
        (posttrained / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    if (
        set(base_index["weight_map"]) != set(post_index["weight_map"])
        or base_index["metadata"]["total_size"] != post_index["metadata"]["total_size"]
    ):
        raise ValueError("Weight tensor roster or total bytes differs")
    base_tokenizer = json.loads((base / "tokenizer.json").read_text(encoding="utf-8"))
    post_tokenizer = json.loads(
        (posttrained / "tokenizer.json").read_text(encoding="utf-8")
    )
    for key in ("vocab", "merges"):
        if base_tokenizer["model"][key] != post_tokenizer["model"][key]:
            raise ValueError(f"Tokenizer {key} differs")
    base_added = {row["content"]: row["id"] for row in base_tokenizer["added_tokens"]}
    post_added = {row["content"]: row["id"] for row in post_tokenizer["added_tokens"]}
    if any(post_added.get(key) != value for key, value in base_added.items()):
        raise ValueError("Existing special-token IDs changed")
    return {
        "config_sha256": sha256_file(base / "config.json"),
        "base_tokenizer_sha256": sha256_file(base / "tokenizer.json"),
        "posttrained_tokenizer_sha256": sha256_file(posttrained / "tokenizer.json"),
        "base_index_sha256": sha256_file(base / "model.safetensors.index.json"),
        "posttrained_index_sha256": sha256_file(
            posttrained / "model.safetensors.index.json"
        ),
        "weight_tensors": len(base_index["weight_map"]),
        "weight_total_bytes": base_index["metadata"]["total_size"],
        "new_special_tokens": dict(sorted(post_added.items() - base_added.items())),
    }


def inspect_weight_headers(model_dir: Path) -> dict[str, Any]:
    """Verify the pinned shard roster and count parameters without loading tensors."""
    from safetensors import safe_open

    index = json.loads(
        (model_dir / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    expected = index["weight_map"]
    seen: dict[str, str] = {}
    parameters = 0
    shard_hashes = {}
    for filename in sorted(set(expected.values())):
        path = model_dir / filename
        shard_hashes[filename] = sha256_file(path)
        with safe_open(path, framework="pt", device="cpu") as tensors:
            names = tensors.keys()
            for name in names:
                if name in seen or expected.get(name) != filename:
                    raise ValueError("Duplicate or mismatched tensor in source shard")
                seen[name] = filename
                parameters += prod(tensors.get_slice(name).get_shape())
    if seen != expected:
        raise ValueError("Source weight shards do not match the index roster")
    return {
        "shard_sha256": shard_hashes,
        "tensor_count": len(seen),
        "safetensors_parameters": parameters,
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    from training.model.decision_model import encode
    from transformers import AutoTokenizer

    metadata = compare_metadata(args.base, args.posttrained)
    source_weights = {
        "base": inspect_weight_headers(args.base),
        "posttrained": inspect_weight_headers(args.posttrained),
    }
    if (
        source_weights["base"]["tensor_count"]
        != source_weights["posttrained"]["tensor_count"]
        or source_weights["base"]["safetensors_parameters"]
        != source_weights["posttrained"]["safetensors_parameters"]
    ):
        raise ValueError("Base and posttrained source weight shapes differ")
    tokenizers = {
        "base": AutoTokenizer.from_pretrained(args.base, local_files_only=True),
        "posttrained": AutoTokenizer.from_pretrained(
            args.posttrained, local_files_only=True
        ),
    }
    token_contract = {
        name: {
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "vocab_size": tokenizer.vocab_size,
            "length": len(tokenizer),
        }
        for name, tokenizer in tokenizers.items()
    }
    counts: dict[str, Any] = {}
    for role, path in (
        ("train", args.train),
        ("select", args.select),
        ("cal", args.cal),
    ):
        rows = load_rows(path, role)
        changed = Counter()
        types = Counter()
        lengths: dict[str, list[int]] = defaultdict(list)
        max_options = 0
        roster = []
        for row in rows:
            old = encode(row, tokenizers["base"], MAX_LENGTH)
            new = encode(row, tokenizers["posttrained"], MAX_LENGTH)
            task_type = row["task_type"]
            types[task_type] += 1
            lengths["base"].append(len(old["ids"]))
            lengths["posttrained"].append(len(new["ids"]))
            max_options = max(max_options, len(old["keys"]))
            if old["prompt_sha256"] != new["prompt_sha256"]:
                raise ValueError("Native rendered prompt changed")
            if old["token_ids_sha256"] != new["token_ids_sha256"]:
                changed[task_type] += 1
            roster.append((row["id"], old["token_ids_sha256"], new["token_ids_sha256"]))
        counts[role] = {
            "rows": len(rows),
            "types": dict(sorted(types.items())),
            "changed_token_ids_by_type": dict(sorted(changed.items())),
            "max_options": max_options,
            "base_tokens": sum(lengths["base"]),
            "posttrained_tokens": sum(lengths["posttrained"]),
            "base_max_length": max(lengths["base"]),
            "posttrained_max_length": max(lengths["posttrained"]),
            "ordered_identity_sha256": canonical_sha256(roster),
        }
    pad_equal = (
        token_contract["base"]["pad_token_id"]
        == token_contract["posttrained"]["pad_token_id"]
    )
    exact_input_parity = pad_equal and all(
        not value["changed_token_ids_by_type"]
        and value["base_tokens"] == value["posttrained_tokens"]
        for value in counts.values()
    )
    return {
        "schema": "decision2-qwen35-4b-posttrained-cpu-input-preflight/1",
        "base_revision": BASE_REVISION,
        "posttrained_revision": POSTTRAINED_REVISION,
        "code_sha256": sha256_file(Path(__file__)),
        "metadata": metadata,
        "source_weights": source_weights,
        "token_contract": token_contract,
        "panels": counts,
        "exact_input_parity": exact_input_parity,
        "scope": "CPU metadata and native-input audit only; no weights loaded or model logits computed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "posttrained", "train", "select", "cal", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps(
            {
                "exact_input_parity": report["exact_input_parity"],
                "rows": {key: value["rows"] for key, value in report["panels"].items()},
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
