"""Threshold-search proposer: bounded perturbation of decision predicates.

Deterministic by construction: the same base config + deltas always produce
the same candidate set (recipe-lab design §Sub-issue 2). Each
candidate is a full copy of the base recipe config with exactly one signal's
predicate threshold moved — one variable per candidate, so a verdict can be
attributed to that variable.

Candidates carry provenance (proposer + the signal they touch) and are plain
YAML config revisions: nothing here talks to a router or a model.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..contract import ContractError


@dataclass(frozen=True)
class ThresholdCandidate:
    signal_name: str
    old_value: float
    new_value: float
    ref: str
    config: dict[str, Any]

    def summary(self) -> str:
        return f"signal {self.signal_name!r} gte {self.old_value} -> {self.new_value}"


def _decision_signal(config: dict[str, Any], signal_name: str) -> dict[str, Any]:
    signals = config.get("routing", {}).get("signals", {}).get("decision", [])
    for signal in signals:
        if isinstance(signal, dict) and signal.get("name") == signal_name:
            return signal
    raise ContractError(f"decision signal {signal_name!r} not found in base config")


def _predicate_value(signal: dict[str, Any]) -> float:
    predicate = signal.get("predicate")
    if not isinstance(predicate, dict) or "gte" not in predicate:
        raise ContractError(
            f"signal {signal.get('name')!r} has no gte predicate to perturb"
        )
    value = predicate["gte"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(
            f"signal {signal.get('name')!r} gte must be a number, got {value!r}"
        )
    return float(value)


def propose_thresholds(
    base_config: dict[str, Any],
    signal_name: str,
    deltas: list[float],
    *,
    ref_prefix: str = "config-rev",
    lo: float = 0.0,
    hi: float = 1.0,
) -> list[ThresholdCandidate]:
    """Generate one candidate per delta, clamped to [lo, hi], baseline dropped."""
    base_signal = _decision_signal(base_config, signal_name)
    old_value = _predicate_value(base_signal)
    candidates: list[ThresholdCandidate] = []
    seen: set[float] = set()
    for delta in deltas:
        new_value = round(min(hi, max(lo, old_value + delta)), 6)
        if new_value == old_value or new_value in seen:
            continue  # the unchanged baseline is not a candidate
        seen.add(new_value)
        config = copy.deepcopy(base_config)
        patched = _decision_signal(config, signal_name)
        patched["predicate"]["gte"] = new_value
        candidates.append(
            ThresholdCandidate(
                signal_name=signal_name,
                old_value=old_value,
                new_value=new_value,
                ref=f"{ref_prefix}:{signal_name}={new_value}",
                config=config,
            )
        )
    return candidates


def write_candidates(
    candidates: list[ThresholdCandidate], output_dir: str | Path
) -> list[Path]:
    """Write each candidate config as YAML plus a sidecar metadata JSON
    (`<stem>.candidate.json`) so `run` can carry the proposer's provenance
    into the ledger; returns the config file paths."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, candidate in enumerate(candidates):
        path = (
            out_dir
            / f"candidate-{index:03d}-{candidate.signal_name}-{candidate.new_value}.yaml"
        )
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(candidate.config, fh, allow_unicode=True, sort_keys=False)
        sidecar = {
            "kind": "config",
            "ref": candidate.ref,
            "change_summary": candidate.summary(),
            "provenance": {"proposer": "threshold-search", "insight_refs": []},
            "sites": [
                {
                    "signal": candidate.signal_name,
                    "op": "gte",
                    "from": candidate.old_value,
                    "to": candidate.new_value,
                }
            ],
        }
        path.with_suffix(".candidate.json").write_text(
            json.dumps(sidecar, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        paths.append(path)
    return paths


DEFAULT_DELTAS = (-0.1, -0.05, 0.05, 0.1)
