"""Frozen TRAIN-only weighting for the matched 4B transfer screen.

The row roster and optimizer schedule stay unchanged. This module contains no
evaluation data, model outputs, or score-dependent tuning.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from training.model.data import digest

PROFILE_VERSION = "eikos4b-human-source-weight-v1"
FROZEN_DATA_SHA256 = {
    "train": "61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755",
    "select": "32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6",
    "cal_audited_only": "3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a",
}
FROZEN_RIGHTS_SHA256 = (
    "61aa883052759830c4ecf897b36c1062ad816c935a12db824c80abd1f80e9ee8"
)
FROZEN_RELEASE_SHA256 = (
    "8978a143290508976d8ddffb735416954cca1de779d5524ca4c59d528539d3dd"
)
HUMAN_SOURCES = frozenset(
    {
        "google_goemotions_official_train",
        "legacy:cosmos_qa",
        "legacy:squad2_answerability",
        "legacy:snli",
        "css_flute_official_train",
    }
)


def loss_weights(
    rows: list[dict[str, Any]], profile: str
) -> tuple[list[float], dict[str, Any]]:
    """Return one weight per admitted row, preserving the original order."""
    if profile not in {"uniform", PROFILE_VERSION}:
        raise ValueError(f"Unsupported Eikos loss profile: {profile}")
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("TRAIN must contain unique, nonempty row IDs")
    counts = Counter(row["source"] for row in rows)
    if profile == PROFILE_VERSION and any(
        source not in counts for source in HUMAN_SOURCES
    ):
        raise ValueError("A frozen human source is absent from admitted TRAIN")
    weights = [
        1.5 if profile == PROFILE_VERSION and row["source"] in HUMAN_SOURCES else 1.0
        for row in rows
    ]
    receipt = {
        "profile": profile,
        "version": PROFILE_VERSION,
        "human_sources": sorted(HUMAN_SOURCES),
        "human_rows": sum(row["source"] in HUMAN_SOURCES for row in rows),
        "score_rows": sum(row["task_type"] == "score" for row in rows),
        "admitted_rows": len(rows),
        "effective_weight_sum": sum(weights),
        "roster_sha256": digest(
            [[row["id"], weight] for row, weight in zip(rows, weights)]
        ),
    }
    return weights, receipt


def verify_frozen_treatment(
    *,
    profile: str,
    args: Any,
    data_hashes: dict[str, str],
    rights_sha256: str,
    release_sha256: str,
    admitted_rows: int,
    admitted_tokens: int,
    receipt: dict[str, Any],
) -> None:
    """Reject a different optimizer, source, schedule or data before loading weights."""
    if profile == "uniform":
        return
    expected_args = {
        "seed": 20260926,
        "max_length": 8192,
        "microbatch": 2,
        "accumulation": 16,
        "eval_batch": 4,
        "max_steps": 232,
        "save_every": 32,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "learning_rate": 2e-5,
        "brier_weight": 0.25,
        "weight_decay": 0.01,
        "select_limit": None,
    }
    actual_args = {name: getattr(args, name) for name in expected_args}
    if actual_args != expected_args:
        raise ValueError("Treatment optimizer or schedule differs from frozen control")
    if data_hashes != FROZEN_DATA_SHA256:
        raise ValueError("Treatment TRAIN/SELECT/CAL bytes differ from frozen control")
    if rights_sha256 != FROZEN_RIGHTS_SHA256 or release_sha256 != FROZEN_RELEASE_SHA256:
        raise ValueError(
            "Treatment rights or upstream release differs from frozen control"
        )
    if (
        admitted_rows != 7418
        or admitted_tokens != 4402743
        or receipt["human_rows"] != 3974
        or receipt["score_rows"] != 516
        or receipt["effective_weight_sum"] != 9405.0
    ):
        raise ValueError("Treatment admitted roster or token budget differs")
