"""Small, strict artifact helpers; endpoint configuration is never copied."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    def reject(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant: {value}")

    return json.loads(path.read_text(), parse_constant=reject)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def finite(value: Any, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("expected a finite number")
    if not math.isfinite(value) or value < minimum:
        raise ValueError("number is nonfinite or below its allowed minimum")
    return float(value)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("JSONL rows must be objects")
    canonical(rows)
    return rows
