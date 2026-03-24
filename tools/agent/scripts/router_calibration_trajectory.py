"""Structured trajectory logging for offline RL training data collection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from router_calibration_support import utc_now, write_json


def _hash_config(config: Any) -> str:
    raw = json.dumps(config, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def save_trajectory(
    report_dir: Path,
    snapshot_before: dict[str, Any],
    pre_eval: dict[str, Any],
    validate_result: dict[str, Any] | None,
    deploy_result: dict[str, Any] | None,
    snapshot_after: dict[str, Any],
    post_eval: dict[str, Any],
) -> None:
    """Save a structured trajectory for offline RL training.

    Tuple format: (theta_t, P, T_t, theta_{t+1}, V(theta_{t+1}), outcome)
    """
    trajectory: dict[str, Any] = {
        "schema_version": 1,
        "timestamp": utc_now(),
        "policy_before": {
            "config_hash": _hash_config(snapshot_before.get("config_classification")),
            "version": snapshot_before.get("config_versions"),
        },
        "observation": {
            "probe_success_rate": pre_eval["success_rate"],
            "decision_success_rate": pre_eval["decision_success_rate"],
            "hybrid_reward": pre_eval.get("hybrid_reward", 0),
            "avg_trace_quality": pre_eval.get("avg_trace_quality", 0),
            "failing_decisions": [
                d["decision_id"]
                for d in pre_eval.get("decisions", [])
                if not d.get("passed")
            ],
            "fragile_decisions": pre_eval.get("fragile_matches", []),
            "per_probe_traces": [
                {
                    "id": r["id"],
                    "matched": r["matched"],
                    "trace_quality": r.get("trace_quality"),
                    "root_cause": r.get("root_cause_classification"),
                }
                for r in pre_eval.get("results", [])
            ],
        },
        "action": {
            "validation_passed": (validate_result or {}).get("valid"),
            "deployed": deploy_result is not None,
            "deploy_version": (deploy_result or {}).get("version"),
            "config_diff_hash": _hash_config(
                snapshot_after.get("config_classification")
            ),
        },
        "outcome": {
            "probe_success_rate": post_eval["success_rate"],
            "decision_success_rate": post_eval["decision_success_rate"],
            "hybrid_reward": post_eval.get("hybrid_reward", 0),
            "avg_trace_quality": post_eval.get("avg_trace_quality", 0),
            "delta_success_rate": (
                post_eval["success_rate"] - pre_eval["success_rate"]
            ),
            "delta_hybrid_reward": (
                post_eval.get("hybrid_reward", 0) - pre_eval.get("hybrid_reward", 0)
            ),
        },
    }
    write_json(report_dir / "trajectory.json", trajectory)
