"""Shared helpers for the Vega data pipeline: JSONL IO, stable hashing, row construction."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import random
import re
import unicodedata
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from d25.vega.common import decision_format as df

WORD = re.compile(r"\w+")
DATA_ROOT = Path(os.environ.get("D25_DATA_ROOT", "/data/d25/vega/data"))


def open_text(path: str | Path, mode: str = "rt"):
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(
            path, mode, encoding=None if "b" in mode else "utf-8", compresslevel=6
        )
    return open(path, mode, encoding=None if "b" in mode else "utf-8")


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open_text(path) as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    """Atomic write (tmp + rename). gzip output is deterministic (mtime 0)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    count = 0
    if str(path).endswith(".gz"):
        with open(tmp, "wb") as raw, gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0
        ) as gz:
            text = io.TextIOWrapper(gz, encoding="utf-8")
            for row in rows:
                text.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                count += 1
            text.flush()
            text.detach()
    else:
        with open(tmp, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                count += 1
    os.replace(tmp, path)
    return count


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(value, indent=1, ensure_ascii=False, sort_keys=False) + "\n"
    )
    os.replace(tmp, path)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def sha(text: str, n: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def rank(key: str, seed: int | str) -> str:
    """Deterministic pseudo-random order key."""
    return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()


def rng_for(key: str, seed: int | str) -> random.Random:
    return random.Random(int(rank(key, seed)[:16], 16))


def normalize_tokens(text: str) -> list[str]:
    """SPEC normalisation: NFKC + casefold + \\w+ tokens."""
    return WORD.findall(unicodedata.normalize("NFKC", text).casefold())


def state_text(state: Any) -> str:
    return df.describe(state) if state not in (None, "") else ""


def option_texts(question: dict[str, Any]) -> list[str]:
    return df.options(question)[1]


def make_row(
    *,
    row_id: str,
    source: str,
    family: str,
    state: Any,
    question: dict[str, Any],
    target: list[float],
    meta: dict[str, Any],
    weight: float = 1.0,
) -> dict[str, Any]:
    total = float(sum(target))
    if total <= 0:
        raise ValueError(f"{row_id}: empty target")
    target = [round(float(v) / total, 6) for v in target]
    drift = 1.0 - sum(target)
    if abs(drift) > 0:
        best = max(range(len(target)), key=target.__getitem__)
        target[best] = round(target[best] + drift, 6)
    label = max(range(len(target)), key=target.__getitem__) if target else -1
    if sorted(target, reverse=True)[:2] == [target[label]] * 2 and len(target) > 1:
        label = -1  # tie: no hard label
    row = {
        "id": row_id,
        "source": source,
        "family": family,
        "state": state,
        "question": question,
        "target": target,
        "label": label,
        "weight": weight,
        "meta": meta,
    }
    df.validate_row(row)
    return row


def choice_question(
    instructions: str | None, criteria: dict[str, Any]
) -> dict[str, Any]:
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def noul_question(instructions: str | None) -> dict[str, Any]:
    return {"type": "noul", "instructions": instructions}
