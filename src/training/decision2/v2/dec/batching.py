"""Token-budget micro-batches and row-count optimizer windows.

Rows are shuffled per epoch from the seed, cut into buckets, sorted by length
inside each bucket and packed greedily into right-padded micro-batches whose
padded size (``max length rounded up to 8 × rows``) stays within
``max_tokens``. Micro-batches are shuffled, then consecutive micro-batches are
accumulated into one optimizer window until it holds at least ``update_rows``
rows. Everything is a pure function of (lengths, seed, epoch), so the planned
update count is known before training.
"""

from __future__ import annotations

import math
import random

BUCKET_ROWS = 4096


def padded(length: int) -> int:
    return math.ceil(length / 8) * 8


def token_batches(
    lengths: list[int], *, seed: int, epoch: int, max_tokens: int, max_rows: int
) -> list[list[int]]:
    if not lengths or max_tokens < 8 or max_rows < 1 or epoch < 0:
        raise ValueError("Need rows, a positive token budget and row cap")
    if padded(max(lengths)) > max_tokens:
        raise ValueError("A single row exceeds the micro-batch token budget")
    rng = random.Random(seed + 1_000_003 * epoch)
    order = list(range(len(lengths)))
    rng.shuffle(order)
    batches: list[list[int]] = []
    for start in range(0, len(order), BUCKET_ROWS):
        bucket = sorted(order[start : start + BUCKET_ROWS], key=lambda i: lengths[i])
        current: list[int] = []
        width = 0
        for index in bucket:
            grown = max(width, padded(lengths[index]))
            if current and (
                grown * (len(current) + 1) > max_tokens or len(current) >= max_rows
            ):
                batches.append(current)
                current, grown = [], padded(lengths[index])
            current.append(index)
            width = grown
        if current:
            batches.append(current)
    rng.shuffle(batches)
    return batches


def row_windows(batches: list[list[int]], update_rows: int) -> list[list[list[int]]]:
    if update_rows < 1:
        raise ValueError("update_rows must be positive")
    windows: list[list[list[int]]] = []
    current: list[list[int]] = []
    rows = 0
    for batch in batches:
        current.append(batch)
        rows += len(batch)
        if rows >= update_rows:
            windows.append(current)
            current, rows = [], 0
    if current:
        windows.append(current)
    return windows
