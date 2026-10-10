"""Sharded, resumable JSONL output shared by the Omni runners.

A row goes to shard ``int(sha256(id)) % num_shards``, so assignment survives reordering and
appended rows. Each shard appends one JSON line per finished row and flushes it; a restart skips
rows already finished (``error`` rows are retried) and drops a torn last line.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def shard_of(key: str, num_shards: int) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % num_shards


def shard_path(out: str | Path, stem: str, shard: int, num_shards: int) -> Path:
    return Path(out) / "shards" / f"{stem}-{shard:03d}-of-{num_shards:03d}.jsonl"


def finished(
    path: Path, retry: Iterable[str] = ("error",)
) -> dict[str, dict[str, Any]]:
    """Complete records of a shard file by id; truncates a torn last line in place."""
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    if text and not text.endswith("\n"):
        text = text[: text.rfind("\n") + 1]
        path.write_text(text, encoding="utf-8")
    records: dict[str, dict[str, Any]] = {}
    for line in text.splitlines():
        if line.strip():
            record = json.loads(line)
            records[record["id"]] = record
    skip = set(retry)
    return {
        key: record
        for key, record in records.items()
        if record.get("status") not in skip
    }


class Appender:
    """Append-only JSONL writer that flushes every record and fsyncs every ``sync_every``."""

    def __init__(self, path: Path, sync_every: int = 64) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("a", encoding="utf-8")
        self.sync_every = sync_every
        self.count = 0

    def write(self, record: dict[str, Any]) -> None:
        self.stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.stream.flush()
        self.count += 1
        if self.count % self.sync_every == 0:
            os.fsync(self.stream.fileno())

    def close(self) -> None:
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()


def merge(out: str | Path, stem: str, num_shards: int) -> dict[str, dict[str, Any]]:
    """Latest record per id over all shard files (all must exist)."""
    records: dict[str, dict[str, Any]] = {}
    for shard in range(num_shards):
        path = shard_path(out, stem, shard, num_shards)
        if not path.exists():
            raise FileNotFoundError(f"missing shard output {path}")
        records.update(finished(path, retry=()))
    return records


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(tmp, target)
