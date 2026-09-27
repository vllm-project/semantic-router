"""Versioned, model-independent Decision Index 0.2.1 panel arithmetic."""

from __future__ import annotations

import hashlib
import json
import math
from importlib.resources import files
from pathlib import Path


def spec() -> dict:
    return json.loads(files(__package__).joinpath("data/protocol-021.json").read_text())


def check_space_bundle(path: str | Path) -> dict:
    """Require the byte-exact Space JSON used when this port was authored."""
    path = Path(path)
    expected = spec()["index_sha256"]
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"Space index bundle SHA-256 mismatch: {actual} != {expected}")
    bundle = json.loads(path.read_text())
    if bundle["suite"]["edition"] != "release-v2.1":
        raise ValueError("Space index bundle has the wrong edition")
    return bundle


def clip(value: float) -> float:
    return max(0.0, min(1.0, value))


def chance_skill(raw: float, chance: float) -> float:
    if not 0 <= chance < 1:
        raise ValueError(f"invalid chance baseline {chance}")
    return clip((raw - chance) / (1 - chance))


def aggregate(
    benchmarks: dict[int | str, dict], *, require_complete: bool = True
) -> dict:
    """Replay 0.2.1's 38-benchmark, five-area index from raw/skill values.

    The caller supplies per-benchmark native metric transformations and
    failure accounting. Four-decimal public values incur at most a small
    display-rounding drift; they do not validate row-level scoring.
    """
    s = spec()
    values = {int(key): value for key, value in benchmarks.items()}
    expected = {number for area in s["areas"] for number in area["benchmarks"]}
    missing = expected - values.keys()
    if require_complete and missing:
        raise ValueError(f"missing headline benchmarks: {sorted(missing)}")
    areas = []
    for area in s["areas"]:
        ids = area["benchmarks"]
        if any(number not in values for number in ids):
            continue
        weights = [
            s["gold_weight"] if number in s["gold_ids"] else 1.0 for number in ids
        ]
        den = sum(weights)

        def mean(field: str, ids=ids, weights=weights, den=den) -> float:
            return (
                sum(
                    weight * values[number][field]
                    for number, weight in zip(ids, weights)
                )
                / den
            )

        areas.append(
            {
                "id": area["id"],
                "weight": area["weight"],
                "raw": mean("raw"),
                "skill": mean("skill"),
                "coverage": mean("coverage"),
                "n": len(ids),
                "benchmarks": list(ids),
            }
        )
    if require_complete and len(areas) != len(s["areas"]):
        raise ValueError("incomplete area coverage")
    scores = {
        "balanced_skill": 100 * sum(a["weight"] * a["skill"] for a in areas),
        "balanced_raw": 100 * sum(a["weight"] * a["raw"] for a in areas),
        "breadth_skill": 100
        * (math.prod((0.1 + 0.9 * a["skill"]) ** a["weight"] for a in areas) - 0.1)
        / 0.9,
    }
    return {
        "edition": s["edition"],
        "panel_id": s["panel_id"],
        "scores": scores,
        "areas": areas,
        "complete": len(areas) == len(s["areas"]),
    }


def replay_published(path: str | Path | None = None) -> dict:
    """Check every public row against its displayed 0.2.1 index arithmetic."""
    fixture = (
        Path(path)
        if path
        else files(__package__).joinpath("data/published-021-summary.json")
    )
    data = json.loads(fixture.read_text())
    if data["source_sha256"] != spec()["index_sha256"]:
        raise ValueError("published fixture is not from the pinned Space bundle")
    deviations = []
    by_model = []
    for row in data["rows"]:
        recomputed = aggregate(row["benchmarks"])
        errors = {
            key: recomputed["scores"][key] - row["scores"][key]
            for key in ("balanced_skill", "balanced_raw", "breadth_skill")
        }
        area_errors = {
            area["id"]: area["skill"] - row["areas"][area["id"]]["skill"]
            for area in recomputed["areas"]
        }
        by_model.append(
            {
                "name": row["name"],
                "score_delta": errors,
                "area_skill_delta": area_errors,
            }
        )
        deviations.extend(abs(v) for v in (*errors.values(), *area_errors.values()))
    return {
        "edition": spec()["edition"],
        "rows": len(by_model),
        "max_display_delta": max(deviations, default=0.0),
        "models": by_model,
        "evidence": "published rounded benchmark aggregates; not row-level prediction parity",
    }
