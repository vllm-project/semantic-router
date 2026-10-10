"""Rows, rendering, targets and the per-update packing plan for the Vega trainer.

Every rank derives the same plan from (seed, epoch, step) alone, so the data cursor is just the
update index and resuming needs no iterator state.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from d25.vega.common import decision_format as fmt


def sha256_file(path: str | Path, chunk: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def input_files(spec: str | Path) -> list[Path]:
    """A JSONL(.gz) file, or the ``train*`` JSONL(.gz) shards of a mixture directory (sorted).

    A directory without ``train*`` shards contributes every JSONL(.gz) except ``dev*`` files.
    """
    path = Path(spec)
    if path.is_dir():
        files = sorted(
            p
            for p in path.iterdir()
            if p.name.endswith((".jsonl", ".jsonl.gz")) and p.is_file()
        )
        train = [p for p in files if p.name.startswith("train")]
        files = train or [p for p in files if not p.name.startswith("dev")]
        if not files:
            raise FileNotFoundError(f"no training .jsonl/.jsonl.gz files in {path}")
        return files
    if not path.exists():
        raise FileNotFoundError(path)
    return [path]


class RowStore:
    """Random access to the rows of one or more JSONL(.gz) files through cached byte offsets.

    Compressed inputs are expanded once into ``cache_dir``; the offset index lives next to it.
    The cache key is the input SHA-256, so concurrent arms share it and edits invalidate it.
    """

    def __init__(self, spec: str | Path, cache_dir: str | Path, *, build: bool):
        self.files = input_files(spec)
        self.cache_dir = Path(cache_dir)
        self.sha256 = {}
        self.plain: list[Path] = []
        offsets = []
        file_ids = []
        for file_id, path in enumerate(self.files):
            plain, index, digest = self._prepare(path, build)
            self.sha256[str(path)] = digest
            self.plain.append(plain)
            values = np.load(index)
            offsets.append(values)
            file_ids.append(np.full(len(values) - 1, file_id, dtype=np.int32))
        self._starts = [values[:-1] for values in offsets]
        self._ends = [values[1:] for values in offsets]
        self.file_of = np.concatenate(file_ids) if file_ids else np.zeros(0, np.int32)
        self.local = (
            np.concatenate([np.arange(len(v) - 1) for v in offsets])
            if offsets
            else np.zeros(0, np.int64)
        )
        self._handles: dict[int, Any] = {}

    def _prepare(self, path: Path, build: bool) -> tuple[Path, Path, str]:
        stamp = f"{path.resolve()}:{path.stat().st_size}:{int(path.stat().st_mtime)}"
        key = hashlib.sha256(stamp.encode()).hexdigest()[:16]
        meta = self.cache_dir / f"{key}.json"
        if meta.exists():
            info = json.loads(meta.read_text())
            return Path(info["plain"]), Path(info["index"]), info["sha256"]
        if not build:
            for _ in range(7200):
                if meta.exists():
                    info = json.loads(meta.read_text())
                    return Path(info["plain"]), Path(info["index"]), info["sha256"]
                time.sleep(1)
            raise TimeoutError(f"row index for {path} was not built")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        digest = sha256_file(path)
        if path.name.endswith(".gz"):
            plain = self.cache_dir / f"{digest}.jsonl"
            if not plain.exists():
                tmp = plain.with_suffix(".tmp")
                with gzip.open(path, "rb") as src, open(tmp, "wb") as dst:
                    while block := src.read(1 << 24):
                        dst.write(block)
                os.replace(tmp, plain)
        else:
            plain = path
        index = self.cache_dir / f"{digest}.offsets.npy"
        if not index.exists():
            starts = [0]
            with open(plain, "rb") as handle:
                position = 0
                for line in handle:
                    position += len(line)
                    starts.append(position)
            # drop blank lines by keeping their spans (they are skipped at read time)
            tmp = index.with_name(index.name + ".tmp.npy")
            np.save(tmp, np.asarray(starts, dtype=np.int64))
            os.replace(tmp, index)
        tmp_meta = meta.with_suffix(".tmp")
        tmp_meta.write_text(
            json.dumps(
                {
                    "plain": str(plain),
                    "index": str(index),
                    "sha256": digest,
                    "source": str(path),
                }
            )
        )
        os.replace(tmp_meta, meta)
        return plain, index, digest

    def __len__(self) -> int:
        return len(self.file_of)

    def raw(self, i: int) -> bytes:
        file_id = int(self.file_of[i])
        j = int(self.local[i])
        handle = self._handles.get(file_id)
        if handle is None:
            handle = open(self.plain[file_id], "rb")
            self._handles[file_id] = handle
        start, end = int(self._starts[file_id][j]), int(self._ends[file_id][j])
        handle.seek(start)
        return handle.read(end - start)

    def get(self, i: int) -> dict[str, Any] | None:
        line = self.raw(i).strip()
        return json.loads(line) if line else None

    def combined_sha256(self) -> str:
        digest = hashlib.sha256()
        for path in self.files:
            digest.update(self.sha256[str(path)].encode())
        return digest.hexdigest()


@dataclass
class Encoded:
    ids: list[int]
    count: int
    target: list[float]
    weight: float
    row_id: str
    kind: str
    family: str


class Encoder:
    """Row -> prompt token ids (decision_format.render) + training target."""

    def __init__(
        self,
        tokenizer,
        codes: list[str],
        *,
        teacher: str | None,
        teacher_weight: float,
        shuffle_options: bool,
        seed: int,
        max_length: int,
    ):
        self.tokenizer = tokenizer
        self.codes = codes
        self.teacher = teacher
        self.teacher_weight = teacher_weight
        self.shuffle_options = shuffle_options
        self.seed = seed
        self.max_length = max_length

    def target(self, row: dict[str, Any]) -> list[float]:
        target = [float(v) for v in row["target"]]
        if self.teacher and self.teacher_weight > 0:
            teachers = (row.get("meta") or {}).get("teachers") or {}
            dist = teachers.get(self.teacher)
            if dist is not None:
                dist = [
                    float(v)
                    for v in (
                        dist.get("probs", dist) if isinstance(dist, dict) else dist
                    )
                ]
                if len(dist) != len(target):
                    raise ValueError(
                        f"row {row['id']}: teacher {self.teacher} has {len(dist)} values for {len(target)} options"
                    )
                total = sum(dist)
                dist = [v / total for v in dist]
                target = [
                    (1 - self.teacher_weight) * t + self.teacher_weight * d
                    for t, d in zip(target, dist)
                ]
        return target

    def encode(self, row: dict[str, Any], *, epoch: int | None, index: int) -> Encoded:
        """``epoch`` None means canonical option order (evaluation)."""
        question = row["question"]
        target = self.target(row)
        ordinal = (row.get("meta") or {}).get("orig_kind") == "score"
        if (
            self.shuffle_options
            and epoch is not None
            and question["type"] == "choice"
            and len(target) > 1
            and not ordinal
        ):
            rng = np.random.default_rng([self.seed, 7919, epoch, index])
            order = rng.permutation(len(target)).tolist()
            items = list(question["criteria"].items())
            question = dict(question)
            question["criteria"] = dict(items[j] for j in order)
            target = [target[j] for j in order]
        text = fmt.render(self.tokenizer, row.get("state"), question, self.codes)
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        weight = float(row.get("weight", 1.0) if row.get("weight") is not None else 1.0)
        return Encoded(
            ids,
            len(target),
            target,
            weight,
            str(row["id"]),
            question["type"],
            str(row.get("family", "")),
        )


def epoch_order(n: int, seed: int, epoch: int) -> np.ndarray:
    return np.random.default_rng([seed, 104729, epoch]).permutation(n)


class Schedule:
    """Global update index -> (epoch, row indices) for ``epochs`` passes in ``rows_per_update`` groups."""

    def __init__(
        self,
        n: int,
        rows_per_update: int,
        epochs: float,
        seed: int,
        skip: np.ndarray | None = None,
    ):
        self.n = n
        self.rows_per_update = rows_per_update
        self.epochs = epochs
        self.seed = seed
        self.keep = np.ones(n, dtype=bool) if skip is None else ~skip
        self.kept = int(self.keep.sum())
        self.per_epoch = math.ceil(self.kept / rows_per_update)
        self.total = max(1, math.ceil(self.per_epoch * epochs - 1e-9))
        self._cache: dict[int, np.ndarray] = {}

    def _order(self, epoch: int) -> np.ndarray:
        if epoch not in self._cache:
            order = epoch_order(self.n, self.seed, epoch)
            self._cache = {epoch: order[self.keep[order]]}
        return self._cache[epoch]

    def rows(self, step: int) -> tuple[int, np.ndarray]:
        """Rows of update ``step`` (0-based)."""
        if not 0 <= step < self.total:
            raise IndexError(step)
        epoch, offset = divmod(step, self.per_epoch)
        order = self._order(epoch)
        return (
            epoch,
            order[offset * self.rows_per_update : (offset + 1) * self.rows_per_update],
        )


def plan_bins(
    lengths: Sequence[int], world: int, budget: int, max_rows: int
) -> list[list[list[int]]]:
    """Pack rows into ``world`` x M micro-batches (same M on every rank), balancing tokens.

    Worst-fit decreasing over ``world * M`` bins of ``budget`` tokens, smallest M that fits; then
    bins go to ranks largest-first onto the least-loaded rank with a free slot. Deterministic.
    Returns ``plan[rank][microbatch] -> positions`` (a micro-batch may be empty).
    """
    if any(n > budget for n in lengths):
        raise ValueError(
            f"a row of {max(lengths)} tokens exceeds the token budget {budget}"
        )
    total = sum(lengths)
    m = max(
        1,
        math.ceil(total / (world * budget)),
        math.ceil(len(lengths) / (world * max_rows)),
    )
    order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))
    while True:
        bins = world * m
        loads = [0] * bins
        members: list[list[int]] = [[] for _ in range(bins)]
        ok = True
        for i in order:
            best = -1
            for b in range(bins):
                if (
                    loads[b] + lengths[i] <= budget
                    and len(members[b]) < max_rows
                    and (best < 0 or loads[b] < loads[best])
                ):
                    best = b
            if best < 0:
                ok = False
                break
            loads[best] += lengths[i]
            members[best].append(i)
        if ok:
            break
        m += 1
    plan: list[list[list[int]]] = [[] for _ in range(world)]
    totals = [0] * world
    for b in sorted(range(bins), key=lambda b: (-loads[b], b)):
        rank = min(
            (r for r in range(world) if len(plan[r]) < m), key=lambda r: (totals[r], r)
        )
        plan[rank].append(sorted(members[b]))
        totals[rank] += loads[b]
    return plan


def length_summary(lengths: Sequence[int]) -> dict[str, float]:
    if not lengths:
        return {}
    values = np.asarray(lengths)
    return {
        "rows": int(values.size),
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
        "max": int(values.max()),
        "tokens": int(values.sum()),
    }


def validate(row: dict[str, Any]) -> str | None:
    """Reason a row cannot be trained on, or None."""
    try:
        fmt.validate_row(row)
    except Exception as exc:  # noqa: BLE001 - reported per row
        return f"invalid: {exc}"
    return None
