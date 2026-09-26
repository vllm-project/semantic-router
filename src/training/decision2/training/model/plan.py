"""Deterministic single-device schedule and exact-resume contract helpers."""

from __future__ import annotations

import math
import random
from typing import Any


def replay_count(train_count: int, pool_count: int, fraction: float) -> int:
    if (
        train_count < 1
        or pool_count < 0
        or not math.isfinite(fraction)
        or not 0 <= fraction < 1
    ):
        raise ValueError("Invalid train/replay counts or replay fraction")
    return min(pool_count, round(train_count * fraction / (1 - fraction)))


def epoch_batches(
    train_lengths: list[int],
    replay_lengths: list[int],
    *,
    epoch: int,
    seed: int,
    microbatch: int,
    replay_fraction: float,
) -> list[list[tuple[str, int]]]:
    if not train_lengths or microbatch < 1 or epoch < 0:
        raise ValueError(
            "Need positive train examples, microbatch, and nonnegative epoch"
        )
    count = replay_count(len(train_lengths), len(replay_lengths), replay_fraction)
    rng = random.Random(seed + 1_000_003 * epoch)
    replay_indices = list(range(len(replay_lengths)))
    rng.shuffle(replay_indices)
    order = [("train", i) for i in range(len(train_lengths))] + [
        ("replay", i) for i in replay_indices[:count]
    ]
    rng.shuffle(order)
    bucket_size = microbatch * 32
    batches: list[list[tuple[str, int]]] = []
    for start in range(0, len(order), bucket_size):
        bucket = order[start : start + bucket_size]
        bucket.sort(
            key=lambda item: (
                train_lengths[item[1]]
                if item[0] == "train"
                else replay_lengths[item[1]]
            )
        )
        local = [bucket[i : i + microbatch] for i in range(0, len(bucket), microbatch)]
        rng.shuffle(local)
        batches.extend(local)
    return batches


def planned_updates(
    train_count: int,
    pool_count: int,
    replay_fraction: float,
    microbatch: int,
    accumulation: int,
    epochs: int,
    max_steps: int | None,
) -> int:
    if (
        microbatch < 1
        or accumulation < 1
        or epochs < 1
        or (max_steps is not None and max_steps < 1)
    ):
        raise ValueError("Invalid batch, accumulation, epoch, or max_steps")
    examples = train_count + replay_count(train_count, pool_count, replay_fraction)
    per_epoch = math.ceil(math.ceil(examples / microbatch) / accumulation)
    return (
        min(per_epoch * epochs, max_steps)
        if max_steps is not None
        else per_epoch * epochs
    )


def validate_resume_state(
    state: dict[str, Any], contract: dict[str, Any], code_hashes: dict[str, str]
) -> None:
    if state.get("contract") != contract:
        raise ValueError(
            "Exact resume requires the same data, model, and optimization contract"
        )
    if state.get("code_sha256") != code_hashes:
        raise ValueError("Exact resume requires unchanged trainer source")
    if (
        type(state.get("step")) is not int
        or not 0 <= state["step"] <= contract["planned_updates"]
    ):
        raise ValueError("Invalid saved optimizer step")
    if (
        type(state.get("next_epoch")) is not int
        or type(state.get("next_batch")) is not int
    ):
        raise ValueError("Invalid saved data cursor")
    if not 0 <= state["next_epoch"] <= contract["epochs"] or state["next_batch"] < 0:
        raise ValueError("Saved data cursor is out of bounds")
    examples = contract["train_count"] + replay_count(
        contract["train_count"],
        contract["replay_pool_count"],
        contract["replay_fraction"],
    )
    batches_per_epoch = math.ceil(examples / contract["microbatch"])
    updates_per_epoch = math.ceil(batches_per_epoch / contract["accumulation"])
    next_batch = state["next_batch"]
    if next_batch >= batches_per_epoch and next_batch != 0:
        raise ValueError("Saved batch cursor exceeds the epoch")
    if next_batch % contract["accumulation"] != 0:
        raise ValueError("Saved batch cursor is not an optimizer boundary")
    expected_step = (
        state["next_epoch"] * updates_per_epoch + next_batch // contract["accumulation"]
    )
    if state["step"] != expected_step:
        raise ValueError("Saved optimizer step and data cursor disagree")
