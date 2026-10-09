"""Paired offline comparison: same calls/budget, no counterfactual E2E timings."""

from __future__ import annotations

import math
import random
from http import HTTPStatus
from itertools import pairwise
from pathlib import Path
from statistics import mean

from .artifacts import digest, file_digest, read_json, read_jsonl, write_json
from .baselines import budget_controls, error_complementarity, random_assignment_control
from .collection import _decode_chat
from .dataset import validate_dataset
from .metrics import evaluate_response, features, summarize
from .policy import (
    calibrated_quality,
    delivery_eligible,
    fit_calibrator,
    fit_heads,
    predict,
    terminal_result,
    unresolved_result,
)
from .timing import COST_METRICS, native_timing

MAX_NATIVE_CALLS = 2


def load_matrix(
    dataset: dict, directory: Path, cost_metric: str = "client_elapsed_ms"
) -> tuple[dict, dict]:
    if cost_metric not in COST_METRICS:
        raise ValueError("unknown cost metric")
    manifest = read_json(directory / "collection.json")
    if manifest["dataset_sha256"] != digest(dataset) or not manifest.get("complete"):
        raise ValueError("replay requires a complete collection of this exact dataset")
    targets = {target["name"]: target for target in manifest["targets"]}
    records = {row["id"]: row for row in dataset["records"]}
    matrix = {key: {} for key in records}
    runtime_identities = {}
    for observation in read_jsonl(directory / "observations.jsonl"):
        identifier, target = observation["record_id"], observation["target"]
        if (
            identifier not in records
            or target not in targets
            or target in matrix[identifier]
        ):
            raise ValueError("unknown or duplicate paired observation")
        row = records[identifier]
        if observation["collection_identity"] != manifest[
            "collection_identity"
        ] or observation["record_request_sha256"] != digest(row["request"]):
            raise ValueError("observation/request identity mismatch")
        if (
            observation["group_id"] != row["group_id"]
            or observation["split"] != row["split"]
        ):
            raise ValueError("observation split/group mismatch")
        elapsed = observation["client_elapsed_ms"]
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(elapsed)
            or elapsed <= 0
        ):
            raise ValueError("elapsed cost must be observed, finite and positive")
        native = targets[target]["protocol"] == "systemone"
        raw = observation["raw_response"]
        response = (
            (raw if native else _decode_chat(raw))
            if HTTPStatus.OK <= observation["http_status"] < HTTPStatus.MULTIPLE_CHOICES
            else {}
        )
        result = evaluate_response(row, response, native=native)
        cost = elapsed
        if native and (cost_metric == "server_compute_ms" or raw.get("meta")):
            compute_cost, identity = native_timing(raw, targets[target])
            if cost_metric == "server_compute_ms":
                cost = compute_cost
            if target in runtime_identities and runtime_identities[target] != identity:
                raise ValueError(
                    "native runtime identity/profile changed within collection"
                )
            runtime_identities[target] = identity
        # Recompute from raw outputs; cached labelled results cannot override ground truth.
        matrix[identifier][target] = {
            **observation,
            "result": result,
            "features": features(row, result) if native else None,
            "policy_cost_ms": cost if native else None,
        }
    if any(set(values) != set(targets) for values in matrix.values()):
        raise ValueError(
            "paired matrix has missing observations; failures must be explicit rows"
        )
    return {**manifest, "runtime_identities": runtime_identities}, matrix


def _risk(observation: dict, calibration: dict) -> float:
    if not delivery_eligible(observation["result"]):
        return 1.0
    return 1 - calibrated_quality(calibration, observation["features"][1])


def choose(
    row: dict, matrix: dict, base: str, setting: dict, policy: dict, calibration: dict
) -> str:
    observation = matrix[row["id"]][base]
    if setting["kind"] == "assignment":
        return setting["actions"][row["id"]]
    if setting["kind"] == "cascade":
        return (
            setting["action"]
            if _risk(observation, calibration) > setting["risk_threshold"]
            else base
        )
    best, value = base, policy["stop_value"]
    for action, head in policy["heads"][base].items():
        gain = predict(head["weights"], observation["features"])
        utility = gain - setting["cost_weight"] * head["training_mean_cost_ms"]
        if utility > value:
            best, value = action, utility
    return best


def execute(
    rows: list[dict],
    matrix: dict,
    base: str,
    setting: dict,
    policy: dict,
    calibration: dict,
) -> list[dict]:
    output = []
    for row in rows:
        action = choose(row, matrix, base, setting, policy, calibration)
        calls = [base] + ([action] if action != base else [])
        source_result = matrix[row["id"]][base]["result"]
        action_result = matrix[row["id"]][action]["result"]
        if setting.get("delivery_kind", setting["kind"]) == "cascade":
            # Reaching the second stage means the first failed its authored
            # acceptance gate. That rejected result cannot become a fallback.
            result = (
                action_result
                if delivery_eligible(action_result)
                else unresolved_result(action_result)
            )
        else:
            result = terminal_result(source_result, action_result)
        output.append(
            {
                "record_id": row["id"],
                "group_id": row["group_id"],
                "action": action,
                "calls": calls,
                "accumulated_observed_call_ms": sum(
                    matrix[row["id"]][name]["client_elapsed_ms"] for name in calls
                ),
                "policy_cost_ms": sum(
                    matrix[row["id"]][name]["policy_cost_ms"] for name in calls
                ),
                "result": result,
            }
        )
    return output


def _aggregate(rows: list[dict]) -> dict:
    return {
        **summarize([row["result"] for row in rows]),
        "mean_accumulated_call_ms": mean(
            row["accumulated_observed_call_ms"] for row in rows
        ),
        "mean_calls": mean(len(row["calls"]) for row in rows),
        "mean_policy_cost_ms": mean(row["policy_cost_ms"] for row in rows),
        "escalation_fraction": mean(len(row["calls"]) > 1 for row in rows),
    }


def select_setting(
    rows: list[dict],
    matrix: dict,
    base: str,
    settings: list[dict],
    budget: float,
    policy: dict,
    calibration: dict,
) -> dict:
    eligible = []
    for setting in settings:
        result = _aggregate(execute(rows, matrix, base, setting, policy, calibration))
        if result["mean_policy_cost_ms"] <= budget + 1e-9:
            eligible.append(
                (
                    result["bundle_error"],
                    result["mean_typed_loss"],
                    result["mean_policy_cost_ms"],
                    digest(setting),
                    setting,
                )
            )
    if not eligible:
        raise ValueError("budget is below mandatory first-stage cost")
    return min(eligible, key=lambda item: item[:4])[4]


def _settings(
    rows: list[dict], matrix: dict, base: str, policy: dict, calibration: dict
) -> tuple[list[dict], list[dict]]:
    thresholds = {
        -1.0,
        1.0,
        *(_risk(matrix[row["id"]][base], calibration) for row in rows),
    }
    cascade = [
        {"kind": "cascade", "action": action, "risk_threshold": threshold}
        for action in policy["heads"][base]
        for threshold in sorted(thresholds)
    ]
    # Breakpoints of gain/cost include every possible policy change in this calibration set.
    crossings = {0.0}
    heads = policy["heads"][base]
    for row in rows:
        x = matrix[row["id"]][base]["features"]
        values = [
            (predict(head["weights"], x), head["training_mean_cost_ms"])
            for head in heads.values()
        ]
        values.append((policy["stop_value"], 0.0))
        for gain, cost in values:
            for other_gain, other_cost in values:
                if cost != other_cost:
                    crossing = (gain - other_gain) / (cost - other_cost)
                    if crossing >= 0:
                        crossings.add(crossing)
    ordered = sorted(crossings)
    # Search the interiors of constant-action regions, not numerical ties.
    # Python and Go may accumulate the same dot product a few ULPs apart.
    weights = {0.0, ordered[-1] + 1.0}
    for left, right in pairwise(ordered):
        middle = left + (right - left) / 2
        if left < middle < right:
            weights.add(middle)
    learned = [{"kind": "policy", "cost_weight": value} for value in sorted(weights)]
    return cascade, learned


def _paired_interval(
    left: list[dict], right: list[dict], seed: int = 42
) -> list[float]:
    """Group bootstrap of policy minus cascade bundle error; pilot uncertainty."""
    groups = {}
    for a, b in zip(left, right, strict=True):
        if a["record_id"] != b["record_id"]:
            raise ValueError("bootstrap comparison must be paired")
        groups.setdefault(a["group_id"], []).append(
            float(b["result"]["correct"]) - float(a["result"]["correct"])
        )
    values = list(groups.values())
    generator = random.Random(seed)
    samples = sorted(
        mean(value for _ in values for value in generator.choice(values))
        for _ in range(1000)
    )
    return [samples[25], samples[974]]


def replay(
    dataset_path: Path,
    collection: Path,
    output: Path,
    base: str,
    *,
    cost_metric: str = "server_compute_ms",
    protocol_path: Path | None = None,
) -> dict:
    dataset = read_json(dataset_path)
    validate_dataset(dataset)
    manifest, matrix = load_matrix(dataset, collection, cost_metric)
    all_native = [
        target["name"]
        for target in manifest["targets"]
        if target["protocol"] == "systemone"
    ]
    native = all_native
    protocol_sha256 = None
    if protocol_path is not None:
        protocol = read_json(protocol_path)
        native = protocol.get("native_pool", [])
        if (
            protocol.get("schema_version") != "systemone-replay-protocol/v1"
            or protocol.get("base") != base
            or protocol.get("dataset_sha256") != digest(dataset)
            or protocol.get("max_model_calls") != MAX_NATIVE_CALLS
            or protocol.get("operating_point_budget_fractions")
            != [0, 0.25, 0.5, 0.75, 1]
            or not isinstance(native, list)
            or len(set(native)) != len(native)
            or any(name not in all_native for name in native)
        ):
            raise ValueError(
                "restricted native-pool protocol does not match this experiment"
            )
        protocol_sha256 = file_digest(protocol_path)
    if base not in native or not any(name != base for name in native):
        raise ValueError(
            "replay requires a native first stage and at least one native alternative"
        )
    public = [row for row in dataset["records"] if row["cohort"] == "public"]
    parts = {
        split: [row for row in public if row["split"] == split]
        for split in ("train", "calibration", "held_out")
    }
    if any(not rows for rows in parts.values()):
        raise ValueError("all three public partitions must be nonempty")
    provenance = {
        "dataset_sha256": digest(dataset),
        "native_pool": native,
        "protocol_sha256": protocol_sha256,
        "trace_sha256": file_digest(collection / "observations.jsonl"),
        "cost_metric": cost_metric,
        "cost_units": "milliseconds",
        "cost_source": (
            "native runtime meta.compute_ms"
            if cost_metric == "server_compute_ms"
            else "serial client perf_counter elapsed"
        ),
    }
    policy = fit_heads(parts["train"], matrix, native, provenance)
    if set(manifest["runtime_identities"]) != set(all_native):
        raise ValueError(
            "offline fitted models require observed runtime identity for every native action"
        )
    policy["actions"] = {
        name: {"model": name, "identity": manifest["runtime_identities"][name]}
        for name in native
    }
    calibration = {
        name: fit_calibrator(
            [
                (
                    matrix[row["id"]][name]["features"][1],
                    matrix[row["id"]][name]["result"]["correct"],
                )
                for row in parts["calibration"]
            ]
        )
        for name in native
    }
    cascade_settings, learned_settings = _settings(
        parts["calibration"], matrix, base, policy, calibration[base]
    )
    mandatory = mean(
        matrix[row["id"]][base]["policy_cost_ms"] for row in parts["calibration"]
    )
    extra = max(
        mean(
            matrix[row["id"]][action]["policy_cost_ms"] for row in parts["calibration"]
        )
        for action in native
        if action != base
    )
    curves, choices = [], []

    def run(rows, setting):
        return execute(rows, matrix, base, setting, policy, calibration[base])

    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        budget = mandatory + fraction * extra
        selected = {
            "cascade": select_setting(
                parts["calibration"],
                matrix,
                base,
                cascade_settings,
                budget,
                policy,
                calibration[base],
            ),
            "policy": select_setting(
                parts["calibration"],
                matrix,
                base,
                learned_settings,
                budget,
                policy,
                calibration[base],
            ),
        }
        held = {
            kind: execute(
                parts["held_out"], matrix, base, setting, policy, calibration[base]
            )
            for kind, setting in selected.items()
        }
        curves.append(
            {
                "calibration_budget_ms": budget,
                "max_calls": MAX_NATIVE_CALLS,
                "settings": selected,
                "calibration": {
                    kind: _aggregate(
                        execute(
                            parts["calibration"],
                            matrix,
                            base,
                            setting,
                            policy,
                            calibration[base],
                        )
                    )
                    for kind, setting in selected.items()
                },
                "held_out": {kind: _aggregate(rows) for kind, rows in held.items()},
                "random_assignment_controls": {
                    kind: random_assignment_control(
                        parts["held_out"], samples, kind, run, _aggregate
                    )
                    for kind, samples in held.items()
                },
                "budget_controls": budget_controls(
                    parts["calibration"],
                    parts["held_out"],
                    matrix,
                    native,
                    base,
                    budget,
                    run,
                    _aggregate,
                ),
                "policy_minus_cascade_bundle_error_group_bootstrap_95pct": _paired_interval(
                    held["policy"], held["cascade"]
                ),
            }
        )
        choices.append({"calibration_budget_ms": budget, **held})
    diagnostics = [row for row in dataset["records"] if row["cohort"] != "public"]
    report = {
        "schema_version": "systemone-replay/v1",
        **provenance,
        "base": base,
        "targets": manifest["targets"],
        "runtime_identities": manifest["runtime_identities"],
        "error_complementarity": {
            split: error_complementarity(parts[split], matrix, native)
            for split in ("train", "calibration")
        },
        "oracle_diagnostic": {
            "native_bundle_accuracy_upper_bound": mean(
                any(matrix[row["id"]][name]["result"]["correct"] for name in native)
                for row in parts["held_out"]
            ),
            "source_groups": len(parts["held_out"]),
            "interpretation": "Label-aware maximum across native model answers; no deployable selector or cost/latency claim",
        },
        "curves": curves,
        "single_model_held_out": {
            name: {
                **summarize(
                    [matrix[row["id"]][name]["result"] for row in parts["held_out"]]
                ),
                "calls_per_request": 1,
                "mean_client_elapsed_ms": mean(
                    matrix[row["id"]][name]["client_elapsed_ms"]
                    for row in parts["held_out"]
                ),
                "mean_policy_cost_ms": (
                    mean(
                        matrix[row["id"]][name]["policy_cost_ms"]
                        for row in parts["held_out"]
                    )
                    if name in all_native
                    else None
                ),
            }
            for name in (target["name"] for target in manifest["targets"])
        },
        "always_escalate_native_controls": {
            name: _aggregate(
                execute(
                    parts["held_out"],
                    matrix,
                    base,
                    {"kind": "cascade", "action": name, "risk_threshold": -1.0},
                    policy,
                    calibration[base],
                )
            )
            for name in native
            if name != base
        },
        "synthetic_diagnostics": {
            name: summarize([matrix[row["id"]][name]["result"] for row in diagnostics])
            for name in native
        },
        "limitations": [
            "Native compute cost is runtime-reported compute elapsed, not kernel GPU-active time or billable provisioned cost",
            "Offline paired replay; additive observed client calls are not routed end-to-end latency",
            "Budget matched on calibration only; held-out realized cost is reported, not constrained after seeing labels",
            "PAVA posterior is estimated bundle correctness, not a certified acceptance guarantee",
            "Qwen direct typed-point baseline is not a deployable judge action and has no confidence metrics",
            "Every fitted head is unconditional on earlier history; this offline experiment uses at most two calls",
            "Native common acceptance requires complete answer evidence, matching top_probability >= 0; valid point-only answers remain valid only in direct baselines",
            "A learned upgrade may retain its previously accepted first answer; an authored cascade cannot fall back to a first answer rejected by its extra acceptance gate",
            "Calibrated unresolved/rejected HTTP503 is not simulated here and must never be dropped from a later online study",
            "No performance gain is presumed; zero/negative improvements must be reported",
        ],
    }
    write_json(output / "fitted-model.json", policy)
    write_json(
        output / "quality-calibration.json",
        {
            "schema_version": "systemone-quality-calibration/v1",
            **provenance,
            "models": calibration,
        },
    )
    write_json(output / "replay.json", report)
    write_json(output / "held-out-decisions.json", choices)
    return report
