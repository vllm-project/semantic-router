"""Seal the existing gold-free typed DEV original/order-variant roster.

This does not construct new questions or inspect targets. The original panel
has four related rows per source group; rows 0 and 2 are the existing original
and order perturbation. Choice/Noul reverse criterion insertion order and
state presentation, while Score preserves ordered levels and perturbs state
presentation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

SOURCE_SHA = "a17ec4b675bbc3da96dba8f31af8f25c9b02cc96ff048fb7de899bdd8b6cf79a"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def seal(source: Path, output: Path) -> dict[str, Any]:
    raw = source.read_bytes()
    if sha(raw) != SOURCE_SHA:
        raise ValueError("Typed DEV prompt source differs from frozen source")
    rows = [json.loads(line) for line in raw.splitlines()]
    if len(rows) != 1600 or output.exists() or output.is_symlink():
        raise ValueError("Expected 1600 source rows and an absent output")
    groups: list[dict[str, Any]] = []
    counts = {"choice": 0, "noul": 0, "score": 0}
    for index in range(0, len(rows), 4):
        quartet = rows[index : index + 4]
        pair = (quartet[0], quartet[2])
        if any(set(row) != {"id", "state", "questions"} for row in quartet):
            raise ValueError(f"Unexpected prompt fields at group {index // 4}")
        if any(set(row["questions"]) != {"decision"} for row in quartet):
            raise ValueError(f"Unexpected question shape at group {index // 4}")
        kinds = {row["questions"]["decision"]["type"] for row in quartet}
        if len(kinds) != 1:
            raise ValueError(f"Mixed question types at group {index // 4}")
        kind = kinds.pop()
        if kind not in counts:
            raise ValueError(f"Unknown type {kind}")
        first, second = (row["questions"]["decision"] for row in pair)
        if first["instructions"] != second["instructions"]:
            raise ValueError(f"Order variant changed instructions at {index // 4}")
        if kind in {"choice", "noul"}:
            keys = list(first["criteria"])
            if list(second["criteria"]) != keys[::-1]:
                raise ValueError(f"Expected reversed keys at group {index // 4}")
            if first["criteria"] != second["criteria"]:
                raise ValueError(f"Order variant changed options at {index // 4}")
        elif first["criteria"] != second["criteria"]:
            raise ValueError(f"Score level order changed at group {index // 4}")
        counts[kind] += 1
        groups.append(
            {
                "group_index": index // 4,
                "type": kind,
                "original_id": pair[0]["id"],
                "order_variant_id": pair[1]["id"],
                "original_input_sha256": sha(
                    json.dumps(
                        pair[0], ensure_ascii=False, separators=(",", ":")
                    ).encode()
                ),
                "order_variant_input_sha256": sha(
                    json.dumps(
                        pair[1], ensure_ascii=False, separators=(",", ":")
                    ).encode()
                ),
            }
        )
    if counts != {"choice": 200, "noul": 100, "score": 100}:
        raise ValueError("Paired roster family counts changed")
    manifest = {
        "schema_version": "decision2-qwen06-paired-order-roster/1",
        "scope": "gold-free existing typed DEV input identities; no target or prediction",
        "source_sha256": SOURCE_SHA,
        "group_count": len(groups),
        "type_group_counts": counts,
        "pairs": groups,
    }
    data = (
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with os.fdopen(
        os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb"
    ) as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return {
        "sha256": sha(data),
        "group_count": len(groups),
        "type_group_counts": counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(seal(args.source, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
