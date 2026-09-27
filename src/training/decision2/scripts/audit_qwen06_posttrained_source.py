"""CPU-only source and tokenizer parity audit for the official 0.6B contrast.

The audit never loads labels for scoring or initializes an optimizer. It
compares the *actual native decision renderer token IDs* on every frozen
TRAIN, SELECT and CAL row, not just the tokenizer vocabulary file names.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from safetensors import safe_open
from training.model.data import (
    canonical,
    check_partition_isolation,
    file_sha256,
    load_partition,
)
from training.model.decision_model import encode
from transformers import AutoTokenizer

BASE_SHA = {
    "config.json": "504a6b58c4271583724e66584b6b7698aea18450209df6b2f7582df0e89cee59",
    "model.safetensors": "cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba",
    "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    "tokenizer_config.json": "3c04ed3ca964ea2f6b2b5faf0dc4d31aec1cb1e8b4bcf63f402d295046b422b5",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "LICENSE": "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e",
}
POST_SHA = {
    "config.json": "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
    "model.safetensors": "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "LICENSE": "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e",
}
PARTITION_SHA = {
    "train": "61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755",
    "select": "32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6",
    "cal": "3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a",
}
PARTITION_ROWS = {"train": 7455, "select": 700, "cal": 700}
BASE_TRAIN_TOKENS = 4094489
MAX_LENGTH = 8192


def _source_parameters(path: Path) -> dict[str, int]:
    with safe_open(path / "model.safetensors", framework="pt", device="cpu") as stream:
        names = list(stream.keys())
        shapes = {name: stream.get_slice(name).get_shape() for name in names}
    counts = {name: int(math.prod(shape)) for name, shape in shapes.items()}
    return {
        "all": sum(counts.values()),
        "lm_head": sum(
            count for name, count in counts.items() if name.startswith("lm_head.")
        ),
        "backbone": sum(
            count for name, count in counts.items() if not name.startswith("lm_head.")
        ),
    }


def audit(
    base: Path,
    posttrained: Path,
    train: Path,
    select: Path,
    cal: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise ValueError("Audit output must not exist")
    for source, expected in ((base, BASE_SHA), (posttrained, POST_SHA)):
        for name, digest in expected.items():
            if file_sha256(source / name) != digest:
                raise ValueError(f"Frozen {source.name} source file differs: {name}")
    for role, path in (("train", train), ("select", select), ("cal", cal)):
        if file_sha256(path) != PARTITION_SHA[role]:
            raise ValueError(f"Frozen {role} partition differs")
    partitions = {
        role: load_partition(path, role)
        for role, path in (("train", train), ("select", select), ("cal", cal))
    }
    check_partition_isolation(partitions)
    if {role: len(rows) for role, rows in partitions.items()} != PARTITION_ROWS:
        raise ValueError("Partition row counts differ")
    configs = [
        json.loads((path / "config.json").read_text()) for path in (base, posttrained)
    ]
    if any(config["model_type"] != "qwen3" for config in configs):
        raise ValueError("A source is not Qwen3")
    if any(config["max_position_embeddings"] < MAX_LENGTH for config in configs):
        raise ValueError("A source is too short for the frozen native renderer")
    if configs[0]["hidden_size"] != configs[1]["hidden_size"]:
        raise ValueError("The readout hidden width differs")
    if b"Apache License" not in (posttrained / "LICENSE").read_bytes():
        raise ValueError("Official source license differs")

    tokenizer_base = AutoTokenizer.from_pretrained(base, local_files_only=True)
    tokenizer_post = AutoTokenizer.from_pretrained(posttrained, local_files_only=True)
    lengths = {role: {"base": 0, "posttrained": 0} for role in partitions}
    mismatches = dict.fromkeys(partitions, 0)
    identities = {}
    token_hashes = {}
    for role, rows in partitions.items():
        identities[role] = hashlib.sha256(
            canonical([row["id"] for row in rows]).encode()
        ).hexdigest()
        hashed = {"base": hashlib.sha256(), "posttrained": hashlib.sha256()}
        for row in rows:
            pair = (
                encode(row, tokenizer_base, MAX_LENGTH),
                encode(row, tokenizer_post, MAX_LENGTH),
            )
            for name, item in zip(("base", "posttrained"), pair):
                lengths[role][name] += len(item["ids"])
                hashed[name].update(
                    canonical({"id": row["id"], "ids": item["ids"]}).encode()
                )
            if (
                pair[0]["ids"] != pair[1]["ids"]
                or pair[0]["candidate_positions"] != pair[1]["candidate_positions"]
                or pair[0]["keys"] != pair[1]["keys"]
            ):
                mismatches[role] += 1
        token_hashes[role] = {name: value.hexdigest() for name, value in hashed.items()}
    if lengths["train"]["base"] != BASE_TRAIN_TOKENS:
        raise ValueError("Base TRAIN token exposure differs from completed control")
    matched = not any(mismatches.values()) and all(
        length["base"] == length["posttrained"] for length in lengths.values()
    )
    result = {
        "schema": "decision2-qwen06-official-posttrained-source-audit/1",
        "status": "PASS" if matched else "HOLD_TOKENIZER_EXPOSURE",
        "base_source": "Qwen/Qwen3-0.6B-Base@da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
        "posttrained_source": "Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca",
        "base_files_sha256": BASE_SHA,
        "posttrained_files_sha256": POST_SHA,
        "partition_sha256": PARTITION_SHA,
        "partition_rows": PARTITION_ROWS,
        "row_order_sha256": identities,
        "input_token_lengths": lengths,
        "input_token_ids_sha256": token_hashes,
        "input_mismatch_rows": mismatches,
        "source_parameters": {
            "base": _source_parameters(base),
            "posttrained": _source_parameters(posttrained),
        },
        "source_config": {
            name: {
                "hidden_size": config["hidden_size"],
                "max_position_embeddings": config["max_position_embeddings"],
                "eos_token_id": config["eos_token_id"],
            }
            for name, config in zip(("base", "posttrained"), configs)
        },
        "code_sha256": file_sha256(Path(__file__)),
        "scope": "CPU identity and exposure only; no model selection or answer scoring",
    }
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    data = (json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n").encode()
    with os.fdopen(
        os.open(output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), "wb"
    ) as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "posttrained", "train", "select", "cal", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        args.base, args.posttrained, args.train, args.select, args.cal, args.output
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "input_mismatch_rows": result["input_mismatch_rows"],
                "input_token_lengths": result["input_token_lengths"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
