"""Template-S training mixtures over hash-pinned data arms.

TRAIN is every row of the base arms plus a token-matched slice of the
treatment arms (experiment matrix v1.1, template S): rho = floor(fraction x
base tokens), capped at the treatment tokens, split across the treatment arms
in proportion to their tokens and filled with whole groups in a fixed hash
order. Tokens are Qwen3-0.6B-Base native tokens of the segmented option prompt
(the data registry's unit), so every testbed trains on the same rows whatever
its own tokenizer.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .common import digest, file_sha256

UNIT = "qwen3-0.6b-base-native-segmented"


def arm_rows(entry: dict[str, Any]) -> list[dict[str, Any]]:
    from training.model.data import load_partition

    path = Path(entry["path"])
    if file_sha256(path) != entry["sha256"]:
        raise ValueError(f"{entry['arm']}: arm file differs from its frozen hash")
    rows = load_partition(path, "train")
    if len(rows) != entry["rows"]:
        raise ValueError(
            f"{entry['arm']}: expected {entry['rows']} rows, found {len(rows)}"
        )
    return rows


def token_counter(tokenizer_path: str) -> Any:
    from transformers import AutoTokenizer

    from training.model.decision_model import encode

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    return lambda row: len(encode(row, tokenizer, 1 << 30)["ids"])


def select_groups(
    rows: list[dict[str, Any]], tokens: list[int], share: int, seed: str, arm: str
) -> list[int]:
    """Whole groups in sha256(seed, arm, group) order, skipping any that would overflow."""
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(row["group_id"], []).append(index)
    order = sorted(
        groups,
        key=lambda g: (hashlib.sha256(f"{seed}\0{arm}\0{g}".encode()).hexdigest(), g),
    )
    chosen, used = [], 0
    for group in order:
        size = sum(tokens[i] for i in groups[group])
        if used + size <= share:
            chosen.extend(groups[group])
            used += size
    return sorted(chosen)


def build(spec: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if spec["template"] != "S" or spec["unit"] != UNIT:
        raise ValueError("Only template S in the Qwen3 native unit is implemented")
    count = token_counter(spec["tokenizer"])
    report: dict[str, Any] = {
        "template": "S",
        "unit": UNIT,
        "seed": spec["seed"],
        "arms": {},
    }
    train: list[dict[str, Any]] = []
    base_tokens = 0
    for entry in spec["base"]:
        rows = arm_rows(entry)
        tokens = sum(count(row) for row in rows)
        base_tokens += tokens
        train.extend(rows)
        report["arms"][entry["arm"]] = {
            "role": "base",
            "rows": len(rows),
            "tokens": tokens,
        }
    treatment = []
    for entry in spec["treatment"]:
        rows = arm_rows(entry)
        treatment.append((entry["arm"], rows, [count(row) for row in rows]))
    available = sum(sum(tokens) for _, _, tokens in treatment)
    rho = min(int(spec["fraction_of_base"] * base_tokens), available)
    for arm, rows, tokens in treatment:
        share = rho * sum(tokens) // available
        chosen = select_groups(rows, tokens, share, spec["seed"], arm)
        picked = [rows[i] for i in chosen]
        train.extend(picked)
        levels: dict[str, int] = {}
        for row in picked:
            if row["task_type"] == "score":
                key = str(len(row["options"]))
                levels[key] = levels.get(key, 0) + 1
        report["arms"][arm] = {
            "role": "treatment",
            "rows_available": len(rows),
            "tokens_available": sum(tokens),
            "share_tokens": share,
            "rows": len(picked),
            "tokens": sum(tokens[i] for i in chosen),
            "groups": len({row["group_id"] for row in picked}),
            "score_level_counts": dict(
                sorted(levels.items(), key=lambda kv: int(kv[0]))
            ),
        }
    ids = [row["id"] for row in train]
    if len(set(ids)) != len(ids):
        raise ValueError("Mixture rows repeat an id across arms")
    report.update(
        {
            "base_tokens": base_tokens,
            "rho_tokens": rho,
            "train_rows": len(train),
            "train_tokens": base_tokens
            + sum(
                a["tokens"] for a in report["arms"].values() if a["role"] == "treatment"
            ),
            "train_ids_sha256": digest(sorted(ids)),
        }
    )
    for key, value in spec.get("expected", {}).items():
        if report[key] != value:
            raise ValueError(f"Mixture {key} is {report[key]}, frozen as {value}")
    return train, report
