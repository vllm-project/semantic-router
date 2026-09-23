"""The append-only judgment store and the choice of what to judge next."""

import json
import os
import random
from pathlib import Path

Records = dict[tuple[str, int], dict]


def read_checkpoint(path: Path) -> Records:
    """Read a checkpoint file into a dict keyed by (split, id).

    Args:
        path: Path to a JSONL checkpoint; a missing file reads as empty.

    Returns:
        The records, where the last record for a key wins.
    """
    out: Records = {}
    if Path(path).exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    out[(record["split"], record["id"])] = record
    return out


def append_records(path: Path, records: list[dict]) -> None:
    """Append records to a checkpoint and fsync, so a crash loses nothing.

    Args:
        path: Path to the JSONL checkpoint.
        records: Records to append.
    """
    with open(path, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def count_judged(records: Records, split: str) -> int:
    """Count the records of one split.

    Args:
        records: Records from read_checkpoint.
        split: Split name.

    Returns:
        The number of rows judged in that split.
    """
    return sum(1 for key in records if key[0] == split)


def records_for_split(records: Records, split: str) -> dict[int, dict]:
    """Pick one split's records, keyed by row id.

    Args:
        records: Records from read_checkpoint.
        split: Split name.

    Returns:
        The split's records by row id.
    """
    return {i: r for (s, i), r in records.items() if s == split}


def select_ids(
    split: str,
    rows: list[dict],
    primary: Records,
    rejudged: Records,
    n: int,
    *,
    rejudge: bool,
    seed: int,
    exclude: frozenset[int] | set[int] = frozenset(),
) -> list[int]:
    """Choose the next row ids to judge.

    Args:
        split: Split name.
        rows: The split's rows.
        primary: Records already in the primary checkpoint.
        rejudged: Records already in the re-judge checkpoint.
        n: Maximum number of ids to return.
        rejudge: Whether to sample rows for the self-consistency pass instead.
        seed: Seed for the re-judge sample.
        exclude: Ids to skip, for example ones that already failed.

    Returns:
        Up to n row ids.
    """
    if rejudge:  # seeded random sample of rows already judged in the primary pass
        pool = sorted(
            i
            for (s, i) in primary
            if s == split and (s, i) not in rejudged and i not in exclude
        )
        random.Random(seed).shuffle(pool)
        return pool[:n]
    return [
        i for i in range(len(rows)) if (split, i) not in primary and i not in exclude
    ][:n]
