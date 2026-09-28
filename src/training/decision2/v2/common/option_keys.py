"""Positional renumbering of construction-order option keys.

Decision 1.0 builders named Choice options `result_<n>` in construction order
and then shuffled the options, so the key number reveals the gold (A7
amendment 1, rule 7d). Keys are model input, so a Choice row whose option keys
all match `result_<n>` is renumbered `result_0` ... `result_{K-1}` by display
position. Option order, descriptions and the label index are unchanged; a
replay row's `teacher_probs` (keyed by option key) moves with its options;
`input_sha256`, when present, is recomputed over the 2.0 input fields. The
original keys and input hash are kept under
`audit_metadata.option_key_renumbering`. Rows with any other key scheme are
returned unchanged, so renumbering is idempotent.

    python3 -m v2.common.option_keys --rows in.jsonl --out out.jsonl --receipt receipt.json

writes the rows in input order (canonical JSON per line) and a count-only
receipt; `--check` writes only the receipt.
"""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import os
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from training.model.data import INPUT_FIELDS, canonical, digest

CONSTRUCTION_ORDER_KEY = re.compile(r"result_(\d+)")
AUDIT_KEY = "option_key_renumbering"
RULE = "positional result_<n> Choice keys (A7 amendment 1, rule 7d)"


def construction_order_keys(row: Mapping[str, Any]) -> list[str] | None:
    """The row's keys if they all match `result_<n>` and are not positional."""
    options = row.get("options")
    if row.get("task_type") != "choice" or not isinstance(options, list) or not options:
        return None
    keys = [
        option.get("key") if isinstance(option, dict) else None for option in options
    ]
    if not all(
        isinstance(key, str) and CONSTRUCTION_ORDER_KEY.fullmatch(key) for key in keys
    ):
        return None
    if keys == positional_keys(len(keys)):
        return None
    return keys


def positional_keys(count: int) -> list[str]:
    return [f"result_{index}" for index in range(count)]


def gold_key_rank(keys: list[str], label: int) -> int:
    """Rank of the gold key's number among the row's key numbers (0 = smallest)."""
    numbers = [int(CONSTRUCTION_ORDER_KEY.fullmatch(key).group(1)) for key in keys]
    return sorted(numbers).index(numbers[label])


def renumber(row: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    """Return (row with positional keys, changed). The input row is not modified."""
    original = construction_order_keys(row)
    if original is None:
        return dict(row), False
    new_keys = positional_keys(len(original))
    out = copy.deepcopy(dict(row))
    out["options"] = [
        dict(option, key=key) for option, key in zip(out["options"], new_keys)
    ]
    teacher = out.get("teacher_probs")
    if isinstance(teacher, dict):
        if set(teacher) != set(original):
            raise ValueError(
                f"{row.get('id')}: teacher_probs keys differ from option keys"
            )
        out["teacher_probs"] = {
            new: teacher[old] for old, new in zip(original, new_keys)
        }
    audit: dict[str, Any] = {"rule": RULE, "original_keys": original}
    if "input_sha256" in out:
        audit["original_input_sha256"] = out["input_sha256"]
        out["input_sha256"] = digest({field: out[field] for field in INPUT_FIELDS})
    metadata = out.get("audit_metadata")
    out["audit_metadata"] = {
        **(metadata if isinstance(metadata, dict) else {}),
        AUDIT_KEY: audit,
    }
    return out, True


def renumber_rows(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Renumber every row; return the rows (input order) and a count-only summary."""
    out: list[dict[str, Any]] = []
    changed: collections.Counter[str] = collections.Counter()
    ranks: collections.Counter[str] = collections.Counter()
    already = 0
    for row in rows:
        keys = construction_order_keys(row)
        new, did = renumber(row)
        if did:
            changed[f"{row.get('task_type')}|{row.get('family')}"] += 1
            rank = gold_key_rank(keys, row["label"])
            where = (
                "largest"
                if rank == len(keys) - 1
                else (
                    "largest_but_one"
                    if rank == len(keys) - 2
                    else "smallest" if rank == 0 else "other"
                )
            )
            ranks[where] += 1
        elif row.get("task_type") == "choice" and [
            option.get("key") for option in row.get("options", [])
        ] == positional_keys(len(row.get("options", []))):
            already += 1
        out.append(new)
    summary = {
        "rows": len(out),
        "renumbered": sum(changed.values()),
        "renumbered_by_type_family": dict(sorted(changed.items())),
        "already_positional_result_keys": already,
        "gold_key_rank_before": dict(sorted(ranks.items())),
        "remaining_construction_order_rows": sum(
            construction_order_keys(row) is not None for row in out
        ),
    }
    return out, summary


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"{path}:{number}: blank line")
            rows.append(json.loads(line))
    return rows


def _write_new(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="receipt only")
    args = parser.parse_args(argv)
    if args.check == (args.out is not None):
        parser.error("give exactly one of --out or --check")
    for path in (args.out, args.receipt):
        if path is not None and path.exists():
            parser.error(f"refusing to overwrite {path}")
    raw = args.rows.read_bytes()
    rows, summary = renumber_rows(_read_jsonl(args.rows))
    receipt = {
        "schema": "decision2.v2.option-key-renumbering.v1",
        "rule": RULE,
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        **summary,
    }
    if args.out is not None:
        data = "".join(canonical(row) + "\n" for row in rows).encode("utf-8")
        _write_new(args.out, data)
        receipt["output_sha256"] = hashlib.sha256(data).hexdigest()
    _write_new(
        args.receipt,
        (json.dumps(receipt, indent=1, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
