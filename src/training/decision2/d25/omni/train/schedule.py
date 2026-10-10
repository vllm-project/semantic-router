"""Step schedule with text replay, and token-budget microbatches per rank.

``build_schedule`` walks the main (multimodal) corpus once per epoch in a seeded shuffle and fills
each optimizer step with ``round(batch * replay_ratio)`` rows from the text replay corpus, drawn
from a seeded stream that reshuffles whenever it is exhausted. ``replay_ratio`` 0 trains on the main
corpus only; 1 trains on the replay corpus only.

``plan_step`` splits one step over the ranks (longest-processing-time on token counts), packs each
rank's rows longest-first into microbatches with ``rows * longest <= token_budget``, and pads every
rank to the same number of microbatches with empty ones, which the trainer runs as zero-weight
dummies so that all ranks issue the same FSDP collectives.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Schedule:
    steps: list[list[int]]
    replay_passes: int
    main_rows: int
    replay_rows: int

    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(self.steps, separators=(",", ":")).encode()
        ).hexdigest()


def build_schedule(
    main: Sequence[int],
    replay: Sequence[int],
    *,
    batch_size: int,
    replay_ratio: float,
    epochs: int = 1,
    seed: int = 0,
) -> Schedule:
    if batch_size < 1 or epochs < 1:
        raise ValueError("batch_size and epochs must be positive")
    if not 0 <= replay_ratio <= 1:
        raise ValueError("replay_ratio must be in [0, 1]")
    if replay_ratio > 0 and not replay:
        raise ValueError("replay_ratio > 0 needs replay rows")
    if replay_ratio < 1 and not main:
        raise ValueError("replay_ratio < 1 needs main rows")
    rng = random.Random(seed)
    primary = list(replay) if replay_ratio == 1 else list(main)
    order: list[int] = []
    for _ in range(epochs):
        epoch = list(primary)
        rng.shuffle(epoch)
        order += epoch
    if replay_ratio in (0, 1):
        steps = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
        return Schedule(
            steps, 0 if replay_ratio == 0 else epochs, len(main), len(replay)
        )
    per_step_main = max(1, batch_size - round(batch_size * replay_ratio))
    stream: list[int] = []
    passes = 0
    steps = []
    for offset in range(0, len(order), per_step_main):
        chunk = order[offset : offset + per_step_main]
        wanted = round(len(chunk) * replay_ratio / (1 - replay_ratio))
        extra: list[int] = []
        while len(extra) < wanted:
            if not stream:
                stream = list(replay)
                rng.shuffle(stream)
                passes += 1
            take = min(wanted - len(extra), len(stream))
            extra += stream[:take]
            stream = stream[take:]
        steps.append(chunk + extra)
    return Schedule(steps, passes, len(main), len(replay))


def plan_step(
    group: Sequence[int],
    tokens: Sequence[int],
    world: int,
    token_budget: int,
    max_rows: int,
) -> list[list[list[int]]]:
    """Per rank, a list of microbatches (row indices); all ranks get the same count."""
    if world < 1 or token_budget < 1 or max_rows < 1:
        raise ValueError("world, token_budget and max_rows must be positive")
    loads = [(0, rank) for rank in range(world)]
    heapq.heapify(loads)
    assigned: list[list[int]] = [[] for _ in range(world)]
    for index in sorted(group, key=lambda i: (-tokens[i], i)):
        load, rank = heapq.heappop(loads)
        assigned[rank].append(index)
        heapq.heappush(loads, (load + tokens[index], rank))
    plans: list[list[list[int]]] = []
    for rows in assigned:
        microbatches: list[list[int]] = []
        current: list[int] = []
        for index in rows:
            if current and (
                len(current) >= max_rows
                or (len(current) + 1) * tokens[current[0]] > token_budget
            ):
                microbatches.append(current)
                current = []
            current.append(index)
        if current:
            microbatches.append(current)
        plans.append(microbatches)
    depth = max(len(microbatches) for microbatches in plans)
    return [
        microbatches + [[] for _ in range(depth - len(microbatches))]
        for microbatches in plans
    ]


def padding_efficiency(
    plans: Sequence[Sequence[Sequence[int]]], tokens: Sequence[int]
) -> float:
    real = sum(tokens[i] for rank in plans for batch in rank for i in batch)
    padded = sum(
        len(batch) * max((tokens[i] for i in batch), default=0)
        for rank in plans
        for batch in rank
    )
    return real / padded if padded else 1.0


def total_steps(schedule: Schedule, stop_after: int | None) -> int:
    return (
        len(schedule.steps)
        if stop_after is None
        else min(len(schedule.steps), stop_after)
    )


def learning_rate_factor(
    step: int, total: int, warmup_ratio: float, floor: float = 0.1
) -> float:
    """Linear warm-up, then cosine decay to ``floor`` (Perplexity's schedule with floor 0.1)."""
    warmup = max(1, int(warmup_ratio * total))
    if step <= warmup:
        return step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))
