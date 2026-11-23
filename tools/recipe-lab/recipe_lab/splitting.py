"""Three-way dataset split (B: calib / dev / frozen_test).

NOTE(verified 2026-10-29): `src/training/model_eval/split_evalset.py` only
implements a 2-way train/eval split of a single manifest (one ratio, no
group-level determinism contract), so recipe-lab implements the minimal
group-level 3-way split here instead of importing it. Determinism contract:
same cases + same group key + same seed => byte-identical output files.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from .contract import SPLIT_NAMES, ContractError, dataset_content_hash

SPLIT_FILES = {
    "calib": "calib.jsonl",
    "dev": "dev.jsonl",
    "frozen_test": "frozen_test.jsonl",
}


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContractError(
                    f"{path}:{line_no} is not valid JSON: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ContractError(f"{path}:{line_no} must be a JSON object")
            if not row.get("case_id"):
                raise ContractError(f"{path}:{line_no} is missing a non-empty case_id")
            if not row.get("prompt"):
                raise ContractError(f"{path}:{line_no} is missing a non-empty prompt")
            cases.append(row)
    if not cases:
        raise ContractError(f"{path} contains no cases")
    return cases


def parse_ratios(raw: str) -> tuple[float, float, float]:
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 3:
        raise ContractError(
            f"--ratios must be three comma-separated numbers, got {raw!r}"
        )
    try:
        values = tuple(float(p) for p in parts)
    except ValueError as exc:
        raise ContractError(f"--ratios must be numbers: {exc}") from exc
    if any(v <= 0 for v in values) or abs(sum(values) - 1.0) > 1e-6:
        raise ContractError(f"--ratios must be positive and sum to 1.0, got {raw!r}")
    return values


def group_cases(
    cases: list[dict[str, Any]], group_key: str
) -> dict[str, list[dict[str, Any]]]:
    """Group cases so one source never spans two splits (leakage guard).
    Cases without the group key fall back to one singleton group per case."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        key = case.get(group_key)
        if not isinstance(key, str) or not key:
            key = f"case:{case['case_id']}"
        groups.setdefault(key, []).append(case)
    return groups


def split_cases(
    cases: list[dict[str, Any]],
    ratios: tuple[float, float, float],
    group_key: str,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    """Deterministic group-level assignment to (calib, dev, frozen_test)."""
    groups = group_cases(cases, group_key)
    names = sorted(groups)  # sort first, then seed-shuffle => deterministic
    rng = random.Random(seed)
    rng.shuffle(names)
    total = len(names)
    n_calib = min(round(total * ratios[0]), total)
    n_dev = min(round(total * ratios[1]), total - n_calib)
    assignment = {
        "calib": names[:n_calib],
        "dev": names[n_calib : n_calib + n_dev],
        "frozen_test": names[n_calib + n_dev :],
    }
    return {
        name: [case for group in assignment[name] for case in groups[group]]
        for name in SPLIT_NAMES
    }


def default_split_rule(
    group_key: str, seed: int, ratios: tuple[float, float, float]
) -> str:
    pcts = [round(r * 100) for r in ratios]
    return f"group by {group_key}; seed {seed}; {pcts[0]}/{pcts[1]}/{pcts[2]}"


def write_splits(
    split_cases_map: dict[str, list[dict[str, Any]]], output_dir: str | Path
) -> dict[str, dict[str, str]]:
    """Write the three JSONL files; returns the spec-ready splits block."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits: dict[str, dict[str, str]] = {}
    for name, split_file in SPLIT_FILES.items():
        target = out_dir / split_file
        with target.open("w", encoding="utf-8") as fh:
            for case in split_cases_map[name]:
                fh.write(json.dumps(case, ensure_ascii=False) + "\n")
        splits[name] = {
            "manifest": split_file,
            "content_hash": dataset_content_hash(target),
        }
    return splits
