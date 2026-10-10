"""Vision-board data: benchmarks, weights, aggregation and live ranks (edition 0.3.1).

The board file is the Space's ``data/vision.json``. Public is ``sum(w_b * max(0, skill_b)) / 9.75``
over 11 benchmarks, private is the plain mean of ``max(0, skill_b)`` over the 9 private sets, and
Full is their mean; ranks are strict (no tie band) and cover entrants only, not reference rows.
"""

from __future__ import annotations

import json
import math
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

LIVE_URL = "https://huggingface.co/spaces/multimodalart/jev-decision-index/resolve/main/data/vision.json"
EDITION = "0.3.1"
WEIGHTS = {
    "CV-Bench": 1.0,
    "BLINK": 1.0,
    "RealWorldQA": 1.0,
    "CharXiv": 1.0,
    "InfographicVQA": 1.0,
    "Mind2Web": 1.0,
    "Winoground": 1.0,
    "KIE (CORD+FUNSD)": 1.0,
    "Moderation (Hateful Memes)": 1.0,
    "R-Bench-M": 0.5,
    "MMMU-Pro vision": 0.25,
}
PUBLIC = tuple(WEIGHTS)
PRIVATE = PUBLIC[:9]
WEIGHT_SUM = sum(WEIGHTS.values())


def clip(value: float) -> float:
    return max(0.0, float(value))


def public_score(skills: Mapping[str, float]) -> float:
    """Board public part from unclipped per-benchmark skills (x100)."""
    return sum(WEIGHTS[b] * clip(skills[b]) for b in PUBLIC) / WEIGHT_SUM


def private_score(skills: Mapping[str, float]) -> float:
    """Board private part: plain mean of the clipped private-set skills."""
    return sum(clip(skills[b]) for b in PRIVATE) / len(PRIVATE)


def full_score(public: float, private: float) -> float:
    return 0.5 * public + 0.5 * private


def load(source: str | Path | None = None, timeout: float = 60.0) -> dict[str, Any]:
    """Board JSON from a local path, an URL, or the live Space when ``source`` is None."""
    target = LIVE_URL if source is None else str(source)
    if target.startswith(("http://", "https://")):
        request = urllib.request.Request(
            target, headers={"User-Agent": "d25-omni-proxy"}
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            board = json.loads(response.read().decode("utf-8"))
    else:
        board = json.loads(Path(target).read_text())
    check_layout(board)
    return board


def check_layout(board: Mapping[str, Any]) -> None:
    weights = board.get("bench_weights") or {}
    if dict(weights) != WEIGHTS:
        raise ValueError(f"board weights changed: {weights}")
    if list(board.get("private_sets") or []) != list(PRIVATE):
        raise ValueError(f"board private sets changed: {board.get('private_sets')}")
    if board.get("weights") != {"public": 0.5, "private": 0.5}:
        raise ValueError(
            f"board public/private weights changed: {board.get('weights')}"
        )


def rows(board: Mapping[str, Any], refs: bool = True) -> list[dict[str, Any]]:
    """Entrant rows (and reference rows when ``refs``), each with per-benchmark ``bench``."""
    out = list(board.get("entrants") or [])
    if refs:
        out += list(board.get("refs") or [])
    return out


def by_engine(board: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["engine"]: row for row in rows(board)}


def per_benchmark(row: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
    """Unclipped public and private skills of one board row."""
    bench = row["bench"]
    public = {b: float(bench[b]["pub"]) for b in PUBLIC}
    private = {b: float(bench[b]["private"]) for b in PRIVATE}
    return public, private


def recompute(row: Mapping[str, Any]) -> dict[str, float]:
    public, private = per_benchmark(row)
    pub = public_score(public)
    priv = private_score(private)
    return {"pub": pub, "priv": priv, "full": full_score(pub, priv)}


def audit(board: Mapping[str, Any], tolerance: float = 0.02) -> list[str]:
    """Rows whose published pub/priv/full differ from the aggregation rule by more than ``tolerance``."""
    problems = []
    for row in rows(board):
        values = recompute(row)
        for key, value in values.items():
            if not math.isclose(value, float(row[key]), abs_tol=tolerance):
                problems.append(
                    f"{row['engine']}: {key} {row[key]} vs recomputed {value:.3f}"
                )
    return problems


def standings(board: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Entrants sorted by Full (strict ranks, ties broken by public as a deterministic fallback)."""
    entrants = sorted(
        rows(board, refs=False),
        key=lambda r: (-float(r["full"]), -float(r["pub"]), r["engine"]),
    )
    return [
        {
            "rank": i + 1,
            "engine": r["engine"],
            "name": r["name"],
            "full": float(r["full"]),
            "pub": float(r["pub"]),
            "priv": float(r["priv"]),
        }
        for i, r in enumerate(entrants)
    ]
