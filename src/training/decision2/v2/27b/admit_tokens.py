"""CPU token admission of training partitions under one source tokenizer.

Uses the native segmented encoder (Qwen) or its BOS-prefixed Gemma variant.
No truncation: a row over the limit is reported, never shortened. Writes only
aggregate counts and digests (no text, no labels).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from training.model.data import canonical, file_sha256
from training.model.decision_model import encode
from training.model.gemma4 import encode_gemma


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def admit(rows: list[dict], tokenizer, encoder, limit: int) -> dict:
    lengths = []
    over = 0
    by_type: dict[str, dict[str, int]] = {}
    for row in rows:
        item = encoder(row, tokenizer, 1 << 20)
        length = len(item["ids"])
        lengths.append([row["id"], length])
        bucket = by_type.setdefault(
            row["task_type"], {"rows": 0, "tokens": 0, "over_limit": 0}
        )
        bucket["rows"] += 1
        bucket["tokens"] += length
        if length > limit:
            over += 1
            bucket["over_limit"] += 1
    values = sorted(length for _, length in lengths)
    return {
        "rows": len(rows),
        "over_limit": over,
        "tokens": sum(values),
        "max_tokens": values[-1],
        "p50_tokens": values[len(values) // 2],
        "p95_tokens": values[int(len(values) * 0.95)],
        "by_task_type": by_type,
        "id_length_digest": hashlib.sha256(
            canonical(lengths).encode("utf-8")
        ).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--family", choices=("qwen", "gemma"), required=True)
    parser.add_argument("--partition", action="append", required=True, help="ROLE=PATH")
    parser.add_argument("--limit", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.source, local_files_only=True)
    encoder = encode if args.family == "qwen" else encode_gemma
    result = {
        "schema_version": "decision2-27b-token-admission/1",
        "tokenizer_json_sha256": file_sha256(args.source / "tokenizer.json"),
        "family": args.family,
        "limit": args.limit,
        "partitions": {},
    }
    for spec in args.partition:
        role, path = spec.split("=", 1)
        path = Path(path)
        result["partitions"][role] = {
            "sha256": file_sha256(path),
            **admit(load_rows(path), tokenizer, encoder, args.limit),
        }
    result["all_admitted"] = all(
        p["over_limit"] == 0 for p in result["partitions"].values()
    )
    fd = os.open(args.output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=1, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps(
            {
                role: {
                    k: v
                    for k, v in p.items()
                    if k in ("rows", "over_limit", "tokens", "max_tokens")
                }
                for role, p in result["partitions"].items()
            }
        )
    )


if __name__ == "__main__":
    main()
