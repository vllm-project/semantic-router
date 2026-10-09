"""Shared helpers for proxy building, running and scoring (kit row format)."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import random
import re
import unicodedata
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

PROXY_VERSION = "pv1"
SALT = "d25-proxy-v1"
SEED = 20261009
WORD = re.compile(r"\w+")

# Board weights for the same-skill aggregate (Space own.json reproduces S exactly with these).
AREA_WEIGHTS = {
    "knowledge": 0.258494,
    "language": 0.258494,
    "retrieval": 0.200229,
    "tools": 0.182783,
    "arts": 0.10,
}
AREAS = {
    "knowledge": [25, 30, 31, 32, 33, 43, 44, 45, 57, 58, 28],
    "language": [11, 12, 29, 38, 39, 40, 41, 42, 59],
    "retrieval": [4, 5, 36, 37, 56, 61],
    "tools": [1, 2, 3, 9, 62],
    "arts": [20, 21, 22, 23, 50, 64],
}
GOLD = {
    57: 1.2,
    58: 1.2,
    25: 1.2,
    45: 1.2,
    12: 1.2,
    28: 1.2,
    29: 1.2,
    4: 1.2,
    5: 1.2,
    36: 1.2,
    1: 1.2,
    3: 1.2,
}
NO_PRIVATE = {11, 45, 23}
NAMES = {
    1: "BFCL",
    2: "ToolRet",
    3: "API-Bank",
    4: "BANKING77",
    5: "CLINC150",
    9: "Home appliances",
    12: "ANLI",
    20: "BPoMP",
    21: "Humicroedit",
    22: "POP909",
    25: "GPQA Diamond",
    28: "WinoGrande",
    29: "HellaSwag",
    30: "GSM8K",
    31: "ChessBench",
    32: "MuSR",
    33: "SATA-Bench",
    36: "BRIGHT",
    37: "Amazon ESCI",
    38: "ACOS",
    39: "FinEntity",
    40: "iSarcasmEval",
    41: "VAST",
    42: "NLI4CT",
    43: "CRUXEval",
    44: "CLadder",
    50: "Habermas",
    56: "PhishNChips",
    57: "MMLU-Pro",
    58: "BBH",
    59: "RAGTruth",
    61: "HoVer",
    62: "When2Call",
    64: "New Yorker",
}
S_BENCHMARKS = sorted(NAMES)


def area_of(n: int) -> str:
    return next(a for a, ids in AREAS.items() if n in ids)


def s_weights() -> dict[int, float]:
    """W_b = area_weight * gold / total gold of the area (all public benchmarks), over the 34 private ones."""
    out = {}
    for area, ids in AREAS.items():
        total = sum(GOLD.get(n, 1.0) for n in ids)
        for n in ids:
            if n not in NO_PRIVATE:
                out[n] = AREA_WEIGHTS[area] * GOLD.get(n, 1.0) / total
    return out


def selected(name: str, key: Any, mod: int, keep: int) -> bool:
    """The holdouts.json slice rule."""
    return (
        int(hashlib.sha256(f"{SALT}:{name}:{key}".encode("utf-8")).hexdigest()[:12], 16)
        % mod
        < keep
    )


def rank(name: str, key: Any) -> str:
    return hashlib.sha256(f"{SALT}:{name}:rank:{key}".encode("utf-8")).hexdigest()


def rng(name: str, key: Any) -> random.Random:
    return random.Random(f"{SALT}:{name}:{key}")


def take(
    items: list,
    n: int,
    name: str,
    key=lambda x: json.dumps(x, sort_keys=True, default=str),
) -> list:
    """Deterministic sample: the n items with the lowest salted hash."""
    return sorted(items, key=lambda x: rank(name, key(x)))[:n]


def stratified(
    items: list,
    n: int,
    name: str,
    stratum,
    key=lambda x: json.dumps(x, sort_keys=True, default=str),
) -> list:
    """Proportional allocation over strata (largest remainder), lowest salted hash inside each stratum."""
    groups: dict[Any, list] = {}
    for x in items:
        groups.setdefault(stratum(x), []).append(x)
    total = len(items)
    if total <= n:
        return list(items)
    share = {k: n * len(v) / total for k, v in groups.items()}
    quota = {k: min(len(groups[k]), int(s)) for k, s in share.items()}
    for k in sorted(groups, key=lambda k: (-(share[k] - int(share[k])), str(k))):
        if sum(quota.values()) >= n:
            break
        if quota[k] < len(groups[k]):
            quota[k] += 1
    out = []
    for k, v in groups.items():
        out += take(v, quota[k], f"{name}:{k}", key)
    return out


def dumps(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def open_text(path: str | Path, mode: str = "rt"):
    path = str(path)
    return (
        gzip.open(path, mode, encoding="utf-8")
        if path.endswith(".gz")
        else open(path, mode.replace("t", ""), encoding="utf-8")
    )


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open_text(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(".partial-" + path.name)
    n = 0
    with open_text(tmp, "wt") as f:
        for r in rows:
            f.write(dumps(r) + "\n")
            n += 1
    os.replace(tmp, path)
    return n


def sha256_file(path: str | Path, gunzip: bool = False) -> str:
    h = hashlib.sha256()
    with gzip.open(path, "rb") if gunzip else open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def norm_tokens(text: str) -> list[str]:
    return WORD.findall(unicodedata.normalize("NFKC", text).casefold())


def norm_text(value: Any) -> str:
    return " ".join(
        norm_tokens(
            value
            if isinstance(value, str)
            else json.dumps(value, ensure_ascii=False, sort_keys=True)
        )
    )


def finish(
    row: dict[str, Any], n: int, track: str | None = None, group: str | None = None
) -> dict[str, Any]:
    """Attach the kit's ``_evaluation`` block so the kit scorers run unchanged on proxy rows."""
    meta = row.setdefault("metadata", {})
    gid = group or meta.get("group_id") or row["id"]
    payload = dumps({"state": row["state"], "questions": row["questions"]})
    row["_evaluation"] = {
        "run_id": f"{PROXY_VERSION}:{n}:{row['id']}",
        "catalog_id": n,
        "dataset": NAMES.get(n, row.get("family")),
        "group_id": str(gid),
        "track": track or row.get("family") or NAMES.get(n),
        "payload_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "proxy_tokens": max(1, len(payload) // 4),
    }
    return row


def mc_row(
    bench: str,
    split: str,
    sid: Any,
    instructions: str,
    options: list[str],
    gold: int,
    state: Any = None,
    **meta,
) -> dict[str, Any]:
    """Kit ``choice_row`` layout (letters up to 26 options, ``option_i`` beyond)."""
    keys = [
        chr(65 + i) if len(options) <= 26 else f"option_{i}"
        for i in range(len(options))
    ]
    return {
        "id": f"{bench}:{split}:{sid}",
        "family": bench,
        "split": split,
        "state": {} if state is None else state,
        "questions": {
            "q1": {
                "type": "choice",
                "instructions": instructions,
                "criteria": dict(zip(keys, options)),
            }
        },
        "expected": {"q1": keys[gold]},
        "provenance": {"source_id": str(sid), **meta},
        "metadata": {"group_id": f"{bench}:{split}:{sid}"},
    }


def question_count(rows: Iterable[dict[str, Any]]) -> int:
    return sum(len(r["questions"]) for r in rows)
