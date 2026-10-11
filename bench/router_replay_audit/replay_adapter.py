#!/usr/bin/env python3
"""Tolerant adapter from vSR Router Replay-shaped JSON into a small audit bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def adapt_replay(record: dict[str, Any]) -> dict[str, Any]:
    """Normalize public Router Replay fields without inventing missing data."""
    route_diag = _dict(record.get("route_diagnostics"))

    decision = record.get("decision")
    if not decision:
        decision = route_diag.get("decision")

    selected_model = record.get("selected_model")
    if not selected_model:
        selected_model = route_diag.get("selected_model")

    bundle = {
        "id": record.get("id"),
        "session_id": record.get("session_id"),
        "turn_index": record.get("turn_index"),
        "decision": decision,
        "selected_model": selected_model,
        "decision_tier": record.get("decision_tier", route_diag.get("decision_tier")),
        "decision_priority": record.get(
            "decision_priority", route_diag.get("decision_priority")
        ),
        "signals": _dict(record.get("signals")),
        "projection_scores": _dict(record.get("projection_scores")),
        "signal_confidences": _dict(record.get("signal_confidences")),
        "signal_values": _dict(record.get("signal_values")),
        "tool_trace": record.get("tool_trace"),
        "session_policy": _dict(record.get("session_policy")),
        "route_diagnostics": route_diag,
        "outcomes": _list(record.get("outcomes")),
        "raw_record": record,
    }
    return bundle


def load_replay(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if "replay" in obj and isinstance(obj["replay"], dict):
        obj = obj["replay"]
    if not isinstance(obj, dict):
        raise ValueError("Replay JSON must be an object")
    return adapt_replay(obj)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("json_file", type=Path)
    args = ap.parse_args()
    print(json.dumps(load_replay(args.json_file), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
