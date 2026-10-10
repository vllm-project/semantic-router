"""Helpers shared by the runtime evidence scripts."""

from __future__ import annotations

import base64
import gzip
import hashlib
import importlib.util
import json
import math
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

CODE_FILES = (
    "decision25_runtime.py",
    "modeling_decision25.py",
    "pipeline_decision25.py",
    "decision25_server.py",
    "decision25_engine.py",
    "requirements.txt",
)
MIME = {"png": "png", "jpg": "jpeg", "jpeg": "jpeg", "webp": "webp"}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, default=str) + "\n", encoding="utf-8")
    tmp.replace(target)


def load_module(name: str, path: Path):
    """Import a flat package file under its own module name (two runtimes can share a process)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_runtime(package: Path, name: str = "decision25_runtime"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))
    return load_module(name, package / "decision25_runtime.py")


def data_url(path: str | Path) -> str:
    path = Path(path)
    kind = MIME[path.suffix.lstrip(".").lower()]
    return f"data:image/{kind};base64," + base64.b64encode(path.read_bytes()).decode()


def sha_key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def stratified(rows: Sequence[dict], n: int, min_multi: int = 0) -> list[dict]:
    """``n`` rows spread evenly over families (sha256 order of the id), plus multi-image rows up to ``min_multi``."""
    families: dict[str, list[dict]] = {}
    for row in rows:
        families.setdefault(row["family"], []).append(row)
    per = math.ceil(n / len(families))
    chosen: list[dict] = []
    for family in sorted(families):
        chosen += sorted(families[family], key=lambda r: sha_key(r["id"]))[:per]
    multi = [r for r in chosen if len(r.get("images") or []) > 1]
    if len(multi) < min_multi:
        ids = {r["id"] for r in chosen}
        extra = sorted(
            (r for r in rows if len(r.get("images") or []) > 1 and r["id"] not in ids),
            key=lambda r: sha_key(r["id"]),
        )
        chosen += extra[: min_multi - len(multi)]
    return chosen


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def latency_stats(milliseconds: Sequence[float]) -> dict[str, float]:
    values = list(milliseconds)
    return {
        "n": len(values),
        "median_ms": round(percentile(values, 0.5), 1),
        "mean_ms": round(sum(values) / len(values), 1),
        "p80_ms": round(percentile(values, 0.8), 1),
        "p95_ms": round(percentile(values, 0.95), 1),
        "max_ms": round(max(values), 1),
    }


def argmax(values: Sequence[float]) -> int:
    return max(range(len(values)), key=values.__getitem__)


def compare_probs(
    pairs: Iterable[tuple[Sequence[float], Sequence[float]]],
) -> dict[str, Any]:
    """Max |dp|, exact matches and argmax agreement over (reference, candidate) probability lists."""
    n = exact = agree = 0
    worst = 0.0
    for want, have in pairs:
        n += 1
        exact += list(want) == list(have)
        agree += argmax(want) == argmax(have)
        worst = max(worst, max(abs(a - b) for a, b in zip(want, have)))
    return {
        "questions": n,
        "exact": exact,
        "argmax_agree": agree,
        "argmax_agreement": round(agree / n, 6) if n else None,
        "max_abs_dp": worst,
    }
