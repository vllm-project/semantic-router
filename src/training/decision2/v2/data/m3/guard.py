"""Refuse teacher-target prompts that touch evaluation, calibration or held-out items (M3a).

    python3 -m v2.data.m3.guard --prompts wave.prompts.jsonl --rows rows.jsonl \\
        --protected-rows select.jsonl ... --protected-prompts typed-final.prompts.jsonl ... \\
        --receipt guard.json

Every prompt must be a TRAIN row of ``--rows`` (``split == "train"``) and equal that
row's native prompt. No prompt may share an id, a row input hash (``input_sha256``) or a
collector prompt digest (state + questions) with any protected row file (SELECT, CAL,
held-out slices) or protected prompt file (gold-free eval panels). Shared normalized
states are counted for the record. The receipt is count-only; exit 3 on any violation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from training.model.data import canonical
from v2.data.build_a0_variants import native_prompt
from v2.data.m2.common import read_jsonl
from v2.data.replay_targets import collector_digest


def state_digest(state: Any) -> str:
    return hashlib.sha256(canonical(state).encode("utf-8")).hexdigest()


def fingerprints(
    protected_rows: list[Path], protected_prompts: list[Path]
) -> tuple[dict[str, set[str]], dict[str, int]]:
    keys: dict[str, set[str]] = {
        "id": set(),
        "input": set(),
        "prompt": set(),
        "state": set(),
    }
    counts: dict[str, int] = {}
    for path in protected_rows:
        n = 0
        for row in read_jsonl(path):
            n += 1
            keys["id"].add(row["id"])
            keys["input"].add(row["input_sha256"])
            keys["prompt"].add(collector_digest(native_prompt(row)))
            keys["state"].add(state_digest(row["state"]))
        counts[path.name] = n
    for path in protected_prompts:
        n = 0
        for prompt in read_jsonl(path):
            n += 1
            keys["id"].add(prompt["id"])
            keys["prompt"].add(collector_digest(prompt))
            keys["state"].add(state_digest(prompt["state"]))
        counts[path.name] = n
    return keys, counts


def guard(
    prompts: Path, rows: Path, protected_rows: list[Path], protected_prompts: list[Path]
) -> dict[str, Any]:
    wanted = {p["id"]: p for p in read_jsonl(prompts)}
    train = {r["id"]: r for r in read_jsonl(rows) if r["id"] in wanted}
    keys, counts = fingerprints(protected_rows, protected_prompts)
    hits = {"id": 0, "input": 0, "prompt": 0, "state": 0}
    not_train = not_row = prompt_mismatch = 0
    for ident, prompt in wanted.items():
        row = train.get(ident)
        if row is None:
            not_row += 1
            continue
        not_train += row.get("split") != "train"
        prompt_mismatch += canonical(prompt) != canonical(native_prompt(row))
        hits["id"] += ident in keys["id"]
        hits["input"] += row["input_sha256"] in keys["input"]
        hits["prompt"] += collector_digest(prompt) in keys["prompt"]
        hits["state"] += state_digest(row["state"]) in keys["state"]
    violations = (
        not_row
        + not_train
        + prompt_mismatch
        + hits["id"]
        + hits["input"]
        + hits["prompt"]
    )
    return {
        "schema": "decision2-m3a-target-guard/1",
        "prompts": len(wanted),
        "prompts_sha256": hashlib.sha256(prompts.read_bytes()).hexdigest(),
        "protected_files": counts,
        "not_a_row": not_row,
        "not_train": not_train,
        "prompt_differs_from_row": prompt_mismatch,
        "shared_id": hits["id"],
        "shared_input_sha256": hits["input"],
        "shared_prompt_digest": hits["prompt"],
        "shared_state_informational": hits["state"],
        "pass": violations == 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--protected-rows", type=Path, action="append", default=[])
    parser.add_argument("--protected-prompts", type=Path, action="append", default=[])
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = guard(
        args.prompts, args.rows, args.protected_rows, args.protected_prompts
    )
    fd = os.open(args.receipt, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=1, sort_keys=True)
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt["pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
