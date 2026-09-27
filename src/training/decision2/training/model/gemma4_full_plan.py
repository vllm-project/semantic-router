"""Pure prospective schedule and development-stop rules for Gemma Decision.

This module never reads benchmark data or starts a device. A full run uses a
private, content-pinned lock generated from TRAIN and SELECT only.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from typing import Any

from .plan import epoch_batches, planned_updates

TRAIN_COUNT = 7287
TRAIN_TOKENS = 3_620_578
TYPE_COUNTS = {"choice": 3804, "noul": 2982, "score": 501}
SELECT_COUNT = 700
SELECT_TYPE_COUNTS = {"choice": 320, "noul": 290, "score": 90}
MAX_LENGTH = 4096
ACCUMULATION = 16
PLANNED_UPDATES = 456
SEED = 20260926
CHECKPOINT_STEPS = (16, 64, 128, 256, 456)
SELECT_STEPS = (64, 128, 256, 456)
MAX_WALL_SECONDS = 86_400
FUTILITY_STEP = 256
FUTILITY_MACRO = 0.70
ADVANCE_MACRO = 0.794
SCORE_MAX_PREDICTED_SHARE = 0.95
SCORE_MIN_PREDICTED_LEVELS = 3


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def admitted_roster(
    rows: list[dict[str, Any]], qwen_lengths: list[int], gemma_lengths: list[int]
) -> tuple[list[int], dict[str, Any]]:
    """Use the previously audited common no-truncation cohort in input order."""
    if len(rows) != len(qwen_lengths) or len(rows) != len(gemma_lengths):
        raise ValueError("Row and token-length vectors differ")
    indices = [
        index
        for index, (qwen, gemma) in enumerate(zip(qwen_lengths, gemma_lengths))
        if 0 < qwen <= MAX_LENGTH and 0 < gemma <= MAX_LENGTH
    ]
    selected = [rows[index] for index in indices]
    types = dict(sorted(Counter(row["task_type"] for row in selected).items()))
    token_total = sum(gemma_lengths[index] for index in indices)
    receipt = {
        "admitted_count": len(indices),
        "admitted_type_counts": types,
        "gemma_unpadded_tokens": token_total,
        "ordered_ids_sha256": digest([row["id"] for row in selected]),
        "ordered_input_hashes_sha256": digest(
            [row["input_sha256"] for row in selected]
        ),
        "gemma_lengths_sha256": digest([gemma_lengths[index] for index in indices]),
    }
    if (
        receipt["admitted_count"] != TRAIN_COUNT
        or types != TYPE_COUNTS
        or token_total != TRAIN_TOKENS
        or receipt["ordered_ids_sha256"]
        != "3d6f96168e38761b45630e9a8a61c60c85b19811df3049e715696acae0edd30b"
    ):
        raise ValueError("Gemma common TRAIN cohort differs from signed CPU audit")
    return indices, receipt


def schedule_receipt(lengths: list[int]) -> dict[str, Any]:
    if len(lengths) != TRAIN_COUNT or sum(lengths) != TRAIN_TOKENS:
        raise ValueError("Gemma TRAIN token lengths differ from signed cohort")
    batches = epoch_batches(
        lengths, [], epoch=0, seed=SEED, microbatch=1, replay_fraction=0.0
    )
    ordered_indices = [index for batch in batches for _, index in batch]
    if (
        len(ordered_indices) != TRAIN_COUNT
        or sorted(ordered_indices) != list(range(TRAIN_COUNT))
        or planned_updates(TRAIN_COUNT, 0, 0.0, 1, ACCUMULATION, 1, None)
        != PLANNED_UPDATES
    ):
        raise ValueError("Gemma one-epoch schedule is incomplete")
    window_tokens = [
        sum(lengths[index] for index in ordered_indices[start : start + ACCUMULATION])
        for start in range(0, TRAIN_COUNT, ACCUMULATION)
    ]
    if len(window_tokens) != PLANNED_UPDATES or sum(window_tokens) != TRAIN_TOKENS:
        raise ValueError("Gemma step token accounting failed")
    return {
        "seed": SEED,
        "microbatch": 1,
        "accumulation": ACCUMULATION,
        "epochs": 1,
        "updates": PLANNED_UPDATES,
        "ordered_row_indices_sha256": digest(ordered_indices),
        "step_token_counts_sha256": digest(window_tokens),
        "unpadded_tokens": TRAIN_TOKENS,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "select_steps": list(SELECT_STEPS),
    }


def select_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Native option argmax and normalized Brier; invalid answer is failure."""
    if len(records) != SELECT_COUNT:
        raise ValueError("SELECT must contain exactly 700 scored rows")
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    score_predictions: Counter[int] = Counter()
    for record in records:
        probs = record["probabilities"]
        gold = record["label"]
        chosen = record["predicted"]
        if (
            not isinstance(probs, list)
            or not 2 <= len(probs) <= 255
            or any(not math.isfinite(p) or p < 0 or p > 1 for p in probs)
            or abs(sum(probs) - 1.0) > 1e-5
            or type(gold) is not int
            or not 0 <= gold < len(probs)
            or type(chosen) is not int
            or not -1 <= chosen < len(probs)
        ):
            raise ValueError("SELECT native probabilities or option label invalid")
        record["correct"] = int(chosen == gold)
        record["brier"] = (
            sum((p - int(index == gold)) ** 2 for index, p in enumerate(probs)) / 2
        )
        by_family[record["family"]].append(record)
        by_type[record["task_type"]].append(record)
        if record["task_type"] == "score" and chosen >= 0:
            score_predictions[chosen] += 1
    if {kind: len(items) for kind, items in by_type.items()} != SELECT_TYPE_COUNTS:
        raise ValueError("SELECT type counts changed")
    family = {
        name: {
            "count": len(items),
            "accuracy": sum(x["correct"] for x in items) / len(items),
            "normalized_brier": sum(x["brier"] for x in items) / len(items),
        }
        for name, items in sorted(by_family.items())
    }
    if not family:
        raise ValueError("SELECT family roster empty")
    return {
        "count": len(records),
        "family_macro_accuracy": sum(x["accuracy"] for x in family.values())
        / len(family),
        "family_macro_brier": sum(x["normalized_brier"] for x in family.values())
        / len(family),
        "by_family": family,
        "by_type": {
            kind: {
                "count": len(items),
                "accuracy": sum(x["correct"] for x in items) / len(items),
            }
            for kind, items in sorted(by_type.items())
        },
        "score_predicted_levels": len(score_predictions),
        "score_max_predicted_share": max(score_predictions.values(), default=0) / 90,
    }


def select_stop_reason(step: int, summary: dict[str, Any]) -> str | None:
    """Predeclared development-only futility and Score collapse thresholds."""
    if step not in SELECT_STEPS:
        raise ValueError("SELECT is allowed only at frozen milestones")
    for key in (
        "family_macro_accuracy",
        "family_macro_brier",
        "score_max_predicted_share",
    ):
        if not math.isfinite(summary[key]):
            return "nonfinite_select_metric"
    if step >= FUTILITY_STEP:
        if summary["family_macro_accuracy"] < FUTILITY_MACRO:
            return "futility_below_0.70"
        if (
            summary["score_predicted_levels"] < SCORE_MIN_PREDICTED_LEVELS
            or summary["score_max_predicted_share"] > SCORE_MAX_PREDICTED_SHARE
        ):
            return "score_prediction_collapse"
    return None


def checkpoint_choice(metrics: dict[int, dict[str, Any]]) -> int:
    if not metrics or any(step not in SELECT_STEPS for step in metrics):
        raise ValueError("Checkpoint choice requires frozen SELECT milestones")
    return max(
        metrics,
        key=lambda step: (
            metrics[step]["family_macro_accuracy"],
            -metrics[step]["family_macro_brier"],
            -step,
        ),
    )
