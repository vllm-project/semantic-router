#!/usr/bin/env python3
"""Fail closed unless a protected run measured all six Decision models.

The report and its raw files are produced during the same protected run as the
candidate image. A report copied from an older commit cannot qualify a release.
This measures synthetic HTTP workflows, not model decision quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from decision_rocm_promotion import MODEL_IDS, validate_receipt
from decision_timed_semantics import validate_timed_semantics

SCHEMA = "decision-paired-release-v7"
RAW_SCHEMA = "decision-semantic-workload-v2"
SOL_GRAPH_MODEL_ID = "llm-semantic-router/Decision-1.0-Sol-2B"
RUNTIME_VARIANTS = frozenset({"eager", "sol_rocm_graph_b8"})
GRAPH_EVENTS = frozenset({"capture", "replay", "fallback"})
ARRIVAL_POLICY = (
    "the same ordered logical workflows are submitted per arm and round; "
    "old fan-out submits one HTTP call per state while new batch submits "
    "one HTTP call per workflow; submission and server admission times "
    "are not paired across arms"
)
SHAPES = ((32, 1), (8, 8), (32, 32))
CONCURRENCIES = (1, 8, 32)
ROUNDS = 3
PHYSICAL_BATCH = 8  # Current untuned qualification ceiling, not a report-wide size.
PHYSICAL_BATCH_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
MIN_WORKFLOWS_PER_ROUND = 32
HIGH_LOAD_SHAPES = ((8, 8), (32, 32))  # Graph replay and batching proof cells.
MIN_HIGH_LOAD_THROUGHPUT_RATIO = 1.00
MAX_SEMANTIC_PROBABILITY_DELTA = 0.010000001
HIGH_LOAD_MIN_CONCURRENCY = 8
SHA = re.compile(r"[0-9a-f]{40}\Z")
HASH = re.compile(r"[0-9a-f]{64}\Z")
IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")
IMAGE_REF = re.compile(r"[^\s@]+@sha256:[0-9a-f]{64}\Z")
MAX_JSON_BYTES = 16 * 1024 * 1024


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"nonfinite JSON number: {value}")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("performance evidence is missing or too large")
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_pairs,
        parse_constant=_reject_nonfinite,
    )
    if not isinstance(value, dict):
        raise ValueError("performance evidence must be a JSON object")
    return value


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _counter_integer(value: Any, label: str, *, minimum: int = 0) -> int:
    """Prometheus emits integral counters as floats in benchmark receipts."""
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or value != math.trunc(value)
        or value < minimum
    ):
        raise ValueError(f"{label} must be an integral counter >= {minimum}")
    return int(value)


def _number(value: Any, label: str, *, positive: bool = True) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    if (positive and value <= 0) or (not positive and value < 0):
        raise ValueError(f"{label} must be nonnegative or positive as required")
    return float(value)


def _hex(value: Any, label: str, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ValueError(f"{label} has an invalid immutable identity")
    return value


def _same(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} differs from protected evidence")


def _close(actual: Any, expected: Any, label: str) -> float:
    a = _number(actual, label)
    b = _number(expected, label)
    if not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"{label} differs from raw measurement")
    return a


def _geometric_mean(ratios: list[float]) -> float:
    return math.exp(math.fsum(math.log(ratio) for ratio in ratios) / len(ratios))


def _clean_mismatches(value: Any, label: str) -> None:
    counts = _mapping(value, label)
    for key, count in counts.items():
        if not isinstance(key, str) or _integer(count, label) != 0:
            raise ValueError(f"{label} reports a semantic mismatch")


def _evidence(root: Path, relative: Any, digest: Any, label: str) -> Path:
    _hex(digest, label + " hash", HASH)
    if not isinstance(relative, str):
        raise ValueError(f"{label} path is invalid")
    name = Path(relative)
    if (
        name.is_absolute()
        or not name.parts
        or name.parts[0] != "raw"
        or ".." in name.parts
        or (
            name.suffix not in (".json", ".jsonl")
            and not name.name.endswith(".jsonl.gz")
        )
    ):
        raise ValueError(f"{label} path must stay under raw/")
    try:
        path = (root / name).resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{label} file is missing") from error
    if not path.is_relative_to(root.resolve(strict=True)) or not path.is_file():
        raise ValueError(f"{label} escapes the evidence directory")
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError(f"{label} is too large")
    with path.open("rb") as handle:
        content = handle.read(MAX_JSON_BYTES + 1)
    if len(content) > MAX_JSON_BYTES:
        raise ValueError(f"{label} is too large")
    if hashlib.sha256(content).hexdigest() != digest:
        raise ValueError(f"{label} content changed")
    return path


def _jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            raise ValueError(f"{label} has a blank record")
        row = json.loads(
            line, object_pairs_hook=_unique_pairs, parse_constant=_reject_nonfinite
        )
        rows.append(_mapping(row, label + " record"))
    if not rows:
        raise ValueError(f"{label} has no records")
    return rows


def _percentiles(values: list[float], raw: Any, label: str) -> None:
    if not values:
        raise ValueError(f"{label} has no measurements")
    ordered = sorted(values)
    summary = _mapping(raw, label + " percentiles")
    for field, fraction in (("p50_ms", 0.5), ("p95_ms", 0.95), ("p99_ms", 0.99)):
        _close(
            summary.get(field),
            ordered[math.ceil(fraction * len(ordered)) - 1],
            label + " " + field,
        )


def _audit_rows(
    path: Path,
    summary: dict[str, Any],
    case_ids: list[str],
    q: int,
    s: int,
    label: str,
) -> dict[str, tuple[tuple[str, ...], str]]:
    rows = _jsonl(path, label)
    _same([row.get("case_id") for row in rows], case_ids, label + " case IDs")
    _same(summary.get("status"), "passed", label + " status")
    _same(summary.get("cases"), len(case_ids), label + " case count")
    _same(summary.get("passed_cases"), len(case_ids), label + " passed cases")
    _same(summary.get("failed_cases"), 0, label + " failed cases")
    _clean_mismatches(summary.get("mismatch_counts"), label + " mismatches")
    tolerance = _number(
        summary.get("probability_tolerance_absolute"),
        label + " probability tolerance",
        positive=False,
    )
    _same(tolerance, 0.01, label + " probability tolerance policy")
    requests = {}
    deltas = []
    for row in rows:
        if (
            row.get("question_count") != q
            or row.get("state_count") != s
            or row.get("passed") is not True
            or row.get("mismatches") != []
            or row.get("request_semantics_identical_except_model_id_and_batch_envelope")
            is not True
        ):
            raise ValueError(f"{label} contains a failed or different case")
        old_http = _list(row.get("old_http"), label + " old HTTP")
        new_http = _mapping(row.get("new_http"), label + " new HTTP")
        if len(old_http) != s:
            raise ValueError(f"{label} old fanout is incomplete")
        hashes = []
        for exchange in [*old_http, new_http]:
            http = _mapping(exchange, label + " HTTP exchange")
            if http.get("status_code") != 200 or http.get("error_code") is not None:
                raise ValueError(f"{label} HTTP exchange failed")
            hashes.append(_hex(http.get("request_sha256"), label + " request", HASH))
            _hex(http.get("response_sha256"), label + " response", HASH)
        states = _list(row.get("states"), label + " states")
        if len(states) != s:
            raise ValueError(f"{label} state inventory is incomplete")
        for state in states:
            state = _mapping(state, label + " state")
            old_tokens = _integer(
                state.get("old_input_tokens"), label + " old input tokens"
            )
            _same(state.get("new_input_tokens"), old_tokens, label + " input tokens")
            _same(state.get("input_token_delta"), 0, label + " input token delta")
            answers = _list(state.get("answers"), label + " answers")
            if len(answers) != q:
                raise ValueError(f"{label} answer inventory is incomplete")
            for answer in answers:
                value = _mapping(answer, label + " answer")
                kind = value.get("type")
                if kind == "noul":
                    old_probability = _number(
                        value.get("old_probability"),
                        label + " old probability",
                        positive=False,
                    )
                    new_probability = _number(
                        value.get("new_probability"),
                        label + " new probability",
                        positive=False,
                    )
                    observed = _number(
                        value.get("absolute_probability_delta"),
                        label + " probability delta",
                        positive=False,
                    )
                    if (
                        old_probability > 1
                        or new_probability > 1
                        or not math.isclose(
                            observed,
                            abs(new_probability - old_probability),
                            rel_tol=1e-9,
                            abs_tol=1e-9,
                        )
                        or observed > tolerance
                    ):
                        raise ValueError(f"{label} noul probability arithmetic differs")
                    deltas.append(observed)
                elif kind in ("choice", "score"):
                    old_probabilities = _mapping(
                        value.get("old_probabilities"), label + " old probabilities"
                    )
                    new_probabilities = _mapping(
                        value.get("new_probabilities"), label + " new probabilities"
                    )
                    distribution = _mapping(
                        value.get("absolute_probability_deltas"),
                        label + " probability deltas",
                    )
                    if (
                        not distribution
                        or set(distribution) != set(old_probabilities)
                        or set(distribution) != set(new_probabilities)
                    ):
                        raise ValueError(f"{label} probability keys differ")
                    for key, observed in distribution.items():
                        delta = _number(
                            observed, label + " probability delta", positive=False
                        )
                        old_probability = _number(
                            old_probabilities[key],
                            label + " old probability",
                            positive=False,
                        )
                        new_probability = _number(
                            new_probabilities[key],
                            label + " new probability",
                            positive=False,
                        )
                        if (
                            old_probability > 1
                            or new_probability > 1
                            or not math.isclose(
                                delta,
                                abs(new_probability - old_probability),
                                rel_tol=1e-9,
                                abs_tol=1e-9,
                            )
                            or delta > tolerance
                        ):
                            raise ValueError(f"{label} probability arithmetic differs")
                        deltas.append(delta)
                    if kind == "choice" and value.get("old_outcome") != value.get(
                        "new_outcome"
                    ):
                        raise ValueError(f"{label} categorical outcomes differ")
                    if kind == "score":
                        old_score = _number(
                            value.get("old_score"), label + " old score", positive=False
                        )
                        new_score = _number(
                            value.get("new_score"), label + " new score", positive=False
                        )
                        if abs(new_score - old_score) > tolerance * (
                            len(distribution) - 1
                        ):
                            raise ValueError(f"{label} score tolerance was exceeded")
                else:
                    raise ValueError(f"{label} answer type is invalid")
        requests[row["case_id"]] = (tuple(hashes[:-1]), hashes[-1])
    if not deltas:
        raise ValueError(f"{label} lacks probability observations")
    delta_summary = _mapping(
        summary.get("absolute_probability_delta"), label + " probability summary"
    )
    _same(delta_summary.get("count"), len(deltas), label + " probability count")
    maximum = _number(
        delta_summary.get("max"), label + " probability max", positive=False
    )
    if not math.isclose(maximum, max(deltas), rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"{label} probability max differs from raw audit")
    return requests


def _raw_arm(
    cell: dict[str, Any], raw: dict[str, Any], label: str, decisions: int
) -> None:
    throughput = _mapping(raw.get("throughput"), label + ".throughput")
    probe = _mapping(raw.get("latency"), label + ".latency")
    rounds = _list(throughput.get("rounds"), label + ".rounds")
    windows = _list(cell.get("round_windows_seconds"), label + ".windows")
    if len(rounds) != ROUNDS or len(windows) != ROUNDS:
        raise ValueError(f"{label} requires exactly three measured rounds")
    for index, (round_, window) in enumerate(zip(rounds, windows, strict=True)):
        _same(round_["round"], index, label + ".round")
        _close(window, round_["window_seconds"], label + ".window")
        if (
            _integer(round_["attempted_workflows"], label + ".round_attempts")
            < MIN_WORKFLOWS_PER_ROUND
        ):
            raise ValueError(f"{label} has too few workflows per round")
        _same(
            round_["successful_workflows"],
            round_["attempted_workflows"],
            label + ".round_success",
        )
        _same(
            round_["attempted_decisions"],
            round_["attempted_workflows"] * decisions,
            label + ".round_decisions",
        )
        _same(
            round_["successful_decisions"],
            round_["attempted_decisions"],
            label + ".round_decision_success",
        )
    for name in (
        "attempted_workflows",
        "successful_workflows",
        "failed_workflows",
        "attempted_decisions",
        "successful_decisions",
    ):
        _same(cell.get(name), throughput.get(name), label + "." + name)
        _integer(cell[name], label + "." + name)
    _same(
        cell["attempted_workflows"],
        sum(r["attempted_workflows"] for r in rounds),
        label + ".attempted_workflows",
    )
    _same(
        cell["attempted_decisions"],
        cell["attempted_workflows"] * decisions,
        label + ".attempted_decisions",
    )
    _same(
        cell["successful_workflows"],
        cell["attempted_workflows"],
        label + ".successful_workflows",
    )
    _same(
        cell["successful_decisions"],
        cell["attempted_decisions"],
        label + ".successful_decisions",
    )
    _same(cell["failed_workflows"], 0, label + ".failed_workflows")
    seconds = sum(_number(window, label + ".window") for window in windows)
    dps = _close(
        cell.get("throughput_decisions_per_second"),
        throughput.get("successful_decisions_per_second"),
        label + ".throughput",
    )
    _close(
        dps, cell["successful_decisions"] / seconds, label + ".throughput_arithmetic"
    )
    for field, raw_section in (
        ("throughput_window_workflow_ms", throughput),
        ("low_load_probe_workflow_ms", probe),
    ):
        latencies = _mapping(cell.get(field), label + "." + field)
        raw_latencies = _mapping(
            raw_section.get("workflow_latency"), label + ".raw_latency"
        )
        for percentile in ("p50_ms", "p95_ms", "p99_ms"):
            _close(
                latencies.get(percentile),
                raw_latencies.get(percentile),
                label + "." + percentile,
            )
        if (
            latencies["p50_ms"] > latencies["p95_ms"]
            or latencies["p95_ms"] > latencies["p99_ms"]
        ):
            raise ValueError(f"{label} has inconsistent latency percentiles")


def _validate_cell(
    cell: dict[str, Any],
    raw: dict[str, Any],
    q: int,
    s: int,
    *,
    same_wire_bytes: bool,
    new_physical_batch_size: int,
) -> None:
    concurrency = cell.get("concurrency")
    _same(raw.get("concurrency"), concurrency, "cell concurrency")
    _same(raw.get("question_count"), q, "cell question count")
    _same(raw.get("state_count"), s, "cell state count")
    summary = _mapping(raw.get("summary"), "raw cell summary")
    comparison = _mapping(summary.get("comparison"), "raw comparison")
    if comparison.get("eligible") is not True or comparison.get("reasons") != []:
        raise ValueError("raw cell comparison is ineligible")
    expected_protocol = (
        "single_fanout_vs_batch_protocol_workflow"
        if s > 1
        else (
            "identical_single_request_bytes"
            if same_wire_bytes
            else "single_request_model_id_adapter_workflow"
        )
    )
    if comparison.get("type") != expected_protocol:
        raise ValueError("raw comparison protocol is mislabeled")
    _same(
        comparison.get("wire_bytes_identical"),
        s == 1 and same_wire_bytes,
        "wire-byte scope",
    )
    arms = _mapping(summary.get("arms"), "raw arms")
    for arm in ("old", "new"):
        _raw_arm(_mapping(cell.get(arm), arm), _mapping(arms.get(arm), arm), arm, q * s)
    old = cell["old"]
    new = cell["new"]
    ratio = _close(
        cell.get("new_over_old_decisions_per_second"),
        comparison.get("new_over_old_successful_decisions_per_second"),
        "throughput ratio",
    )
    _close(
        ratio,
        new["throughput_decisions_per_second"] / old["throughput_decisions_per_second"],
        "throughput ratio arithmetic",
    )
    latency_ratio = _close(
        cell.get("old_over_new_low_load_p50_ms"),
        comparison.get("old_over_new_p50_workflow_latency"),
        "low-load latency ratio",
    )
    _close(
        latency_ratio,
        old["low_load_probe_workflow_ms"]["p50_ms"]
        / new["low_load_probe_workflow_ms"]["p50_ms"],
        "low-load ratio arithmetic",
    )
    telemetry = _mapping(
        _mapping(summary.get("telemetry"), "telemetry").get("arms"), "telemetry arms"
    )
    measured = _mapping(telemetry.get("new"), "new physical batch telemetry")
    if (
        measured.get("status") != "complete"
        or _integer(measured.get("rounds"), "telemetry rounds") != ROUNDS
    ):
        raise ValueError("new physical batch telemetry is incomplete")
    counters = _mapping(measured.get("counter_deltas"), "new physical batch counters")
    batches = _counter_integer(
        cell.get("new_physical_batches"), "new physical batches", minimum=1
    )
    rows = _counter_integer(
        cell.get("new_physical_batch_rows"), "new physical batch rows", minimum=1
    )
    _same(batches, counters.get("physical_batches"), "physical batches")
    _same(rows, counters.get("physical_batch_rows"), "physical batch rows")
    graph_events = _mapping(measured.get("graph_event_deltas"), "new graph events")
    _same(
        _counter_integer(cell.get("new_graph_replays"), "new graph replays"),
        _counter_integer(graph_events.get("replay"), "raw new graph replays"),
        "new graph replay count",
    )
    if rows < batches or rows > batches * new_physical_batch_size:
        raise ValueError("observed physical batch rows exceed declared batch capacity")
    observed = _close(
        cell.get("new_observed_rows_per_physical_batch"),
        measured.get("observed_rows_per_physical_batch"),
        "observed physical batch size",
    )
    _close(observed, rows / batches, "physical batch arithmetic")


def _validate_workflows(
    path: Path,
    raw_rows: list[dict[str, Any]],
    q: int,
    s: int,
    case_ids: list[str],
    settings: dict[str, Any],
) -> list[dict[str, Any]]:
    """Verify the three alternating, nonoverlapping old/new throughput waves."""
    workflows = _jsonl(path, "workflows")
    paired_cases: dict[tuple[int, str, int, int], dict[str, str]] = defaultdict(dict)
    counts: Counter[tuple[int, str, str]] = Counter()
    for record in workflows:
        arm = record.get("arm")
        phase = record.get("phase")
        concurrency = record.get("concurrency")
        round_number = record.get("round")
        sequence = record.get("sequence")
        case_id = record.get("case_id")
        if (
            arm not in ("old", "new")
            or phase not in ("warmup", "latency", "throughput")
            or concurrency not in CONCURRENCIES
            or type(round_number) is not int
            or round_number < 0
            or type(sequence) is not int
            or sequence < 0
            or case_id not in case_ids
            or arm in paired_cases[(concurrency, phase, round_number, sequence)]
        ):
            raise ValueError("workflow identity or case ID is invalid")
        paired_cases[(concurrency, phase, round_number, sequence)][arm] = case_id
        counts[(concurrency, phase, arm)] += 1
    if any(
        set(pair) != {"old", "new"} or pair["old"] != pair["new"]
        for pair in paired_cases.values()
    ):
        raise ValueError("old/new workflow case IDs differ")
    for concurrency in CONCURRENCIES:
        for phase, expected in (
            ("warmup", settings["warmup_workflows_per_arm"]),
            ("latency", settings["latency_workflows_per_arm"]),
            ("throughput", settings["throughput_workflows_per_round_per_arm"] * ROUNDS),
        ):
            for arm in ("old", "new"):
                if counts[(concurrency, phase, arm)] != expected:
                    raise ValueError("workflow phase inventory is incomplete")
            if phase != "throughput" and {
                (round_number, sequence)
                for cell, recorded_phase, round_number, sequence in paired_cases
                if cell == concurrency and recorded_phase == phase
            } != {(index, 0) for index in range(expected)}:
                raise ValueError("low-load workflow schedule is incomplete")
    expected_count = 0
    previous_cell_end: float | None = None
    for raw in raw_rows:
        concurrency = raw["concurrency"]
        cell_workflows = [
            record for record in workflows if record["concurrency"] == concurrency
        ]
        cell_start = min(
            _number(record.get("started_offset_ms"), "cell start", positive=False)
            for record in cell_workflows
        )
        cell_end = max(
            _number(record.get("completed_offset_ms"), "cell end")
            for record in cell_workflows
        )
        if previous_cell_end is not None and cell_start < previous_cell_end:
            raise ValueError("concurrency cells overlap or run out of order")
        previous_cell_end = cell_end
        summary = raw["summary"]
        # The producer runs each low-load probe serially, alternating the arm
        # order, then starts the measured throughput waves. Allowing a probe
        # to overlap a c1 wave would silently add load to one baseline arm.
        last_end: float | None = None
        for phase, count in (
            ("warmup", settings["warmup_workflows_per_arm"]),
            ("latency", settings["latency_workflows_per_arm"]),
        ):
            for sequence in range(count):
                pair = {
                    record["arm"]: record
                    for record in cell_workflows
                    if record["phase"] == phase and record["round"] == sequence
                }
                if set(pair) != {"old", "new"}:
                    raise ValueError("low-load arm inventory is incomplete")
                order = ("old", "new") if sequence % 2 == 0 else ("new", "old")
                for arm in order:
                    record = pair[arm]
                    start = _number(
                        record.get("started_offset_ms"),
                        "low-load workflow start",
                        positive=False,
                    )
                    end = _number(
                        record.get("completed_offset_ms"),
                        "low-load workflow end",
                    )
                    if end <= start or (last_end is not None and start < last_end):
                        raise ValueError("low-load phases or arms overlap")
                    last_end = end
        waves = {}
        for arm in ("old", "new"):
            rounds = summary["arms"][arm]["throughput"]["rounds"]
            measured_latencies = []
            for index, round_ in enumerate(rounds):
                selected = [
                    record
                    for record in workflows
                    if record.get("phase") == "throughput"
                    and record.get("arm") == arm
                    and record.get("round") == index
                    and record.get("concurrency") == concurrency
                ]
                count = round_["attempted_workflows"]
                if len(selected) != count or {
                    record.get("sequence") for record in selected
                } != set(range(count)):
                    raise ValueError("throughput workflow inventory is incomplete")
                expected_count += count
                for record in selected:
                    if (
                        record.get("success") is not True
                        or record.get("error_codes") != []
                    ):
                        raise ValueError("throughput workflow failed")
                    _same(record.get("decisions"), q * s, "workflow decisions")
                    _same(
                        record.get("http_calls"),
                        s if arm == "old" else 1,
                        "workflow HTTP calls",
                    )
                    start = _number(
                        record.get("started_offset_ms"),
                        "workflow start",
                        positive=False,
                    )
                    end = _number(record.get("completed_offset_ms"), "workflow end")
                    if end <= start:
                        raise ValueError("workflow has nonpositive duration")
                    latency = _number(record.get("latency_ms"), "workflow latency")
                    if not math.isclose(
                        latency, end - start, rel_tol=1e-6, abs_tol=1e-6
                    ):
                        raise ValueError("workflow latency differs from its timestamps")
                    measured_latencies.append(latency)
                start = min(record["started_offset_ms"] for record in selected)
                end = max(record["completed_offset_ms"] for record in selected)
                if not math.isclose(
                    (end - start) / 1000,
                    round_["window_seconds"],
                    rel_tol=1e-6,
                    abs_tol=1e-6,
                ):
                    raise ValueError(
                        "throughput wave duration differs from raw workflows"
                    )
                waves[index, arm] = (start, end)
            latency_summary = summary["arms"][arm]["throughput"]["workflow_latency"]
            measured_latencies.sort()
            for field, fraction in (
                ("p50_ms", 0.50),
                ("p95_ms", 0.95),
                ("p99_ms", 0.99),
            ):
                nearest = measured_latencies[
                    math.ceil(fraction * len(measured_latencies)) - 1
                ]
                _close(
                    latency_summary[field], nearest, "throughput workflow percentile"
                )
        for index in range(ROUNDS):
            first, second = ("old", "new") if index % 2 == 0 else ("new", "old")
            first_start, first_end = waves[index, first]
            second_start, second_end = waves[index, second]
            if (
                last_end is not None and first_start < last_end
            ) or second_start < first_end:
                raise ValueError(
                    "throughput arms were not measured in alternating waves"
                )
            last_end = second_end
    if (
        sum(record.get("phase") == "throughput" for record in workflows)
        != expected_count
    ):
        raise ValueError("unexpected throughput workflow records")
    return workflows


def _validate_samples(
    path: Path,
    workflows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    requests: dict[str, tuple[tuple[str, ...], str]],
    q: int,
    s: int,
) -> list[dict[str, Any]]:
    samples = _jsonl(path, "samples")
    intervals: dict[tuple[Any, ...], list[tuple[float, int]]] = defaultdict(list)
    for sample in samples:
        concurrency = sample.get("concurrency")
        if concurrency not in CONCURRENCIES:
            raise ValueError("sample concurrency is invalid")
        start = _number(sample.get("started_offset_ms"), "sample start", positive=False)
        end = _number(sample.get("completed_offset_ms"), "sample end")
        if end <= start:
            raise ValueError("sample interval is invalid")
        phase = sample.get("phase")
        # Low-load requests run sequentially across workflow sequence numbers.
        # Keeping the sequence as a separate group hides a c1 burst of 16.
        round_number = sample.get("round") if phase == "throughput" else None
        key = (concurrency, sample.get("arm"), phase, round_number)
        intervals[key].extend(((start, 1), (end, -1)))
    for (concurrency, _arm, _phase, _round), events in intervals.items():
        active = 0
        for _time, change in sorted(events):  # -1 ends before +1 starts on ties.
            active += change
            if active < 0 or active > concurrency:
                raise ValueError(
                    "sample peak in-flight HTTP exceeds declared concurrency"
                )
    schedules: dict[tuple[int, str, int], dict[int, str]] = defaultdict(dict)
    for workflow in workflows:
        if workflow["arm"] == "old":
            schedules[(workflow["concurrency"], workflow["phase"], workflow["round"])][
                workflow["sequence"]
            ] = workflow["case_id"]
    schedule_hashes = {}
    for (concurrency, phase, round_number), cases in schedules.items():
        if set(cases) != set(range(len(cases))):
            raise ValueError("logical schedule sequence is incomplete")
        payload = {
            "question_count": q,
            "state_count": s,
            "concurrency": concurrency,
            "phase": phase,
            "round": round_number,
            "case_ids": [cases[index] for index in range(len(cases))],
        }
        schedule_hashes[(concurrency, phase, round_number)] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def key(record: dict[str, Any]) -> tuple[Any, ...]:
        return tuple(
            record.get(field)
            for field in ("concurrency", "arm", "phase", "round", "sequence")
        )

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[key(sample)].append(sample)
    workflow_keys = {key(record) for record in workflows}
    if set(grouped) != workflow_keys or len(workflow_keys) != len(workflows):
        raise ValueError("sample/workflow identity inventory differs")
    by_cell_phase: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for workflow in workflows:
        arm = workflow["arm"]
        if (
            workflow.get("success") is not True
            or workflow.get("error_codes") != []
            or workflow.get("decisions") != q * s
            or workflow.get("http_calls") != (s if arm == "old" else 1)
        ):
            raise ValueError("raw workflow has a failure or different work count")
        workflow_start = _number(
            workflow.get("started_offset_ms"), "workflow start", positive=False
        )
        workflow_end = _number(workflow.get("completed_offset_ms"), "workflow end")
        workflow_latency = _number(workflow.get("latency_ms"), "workflow latency")
        if workflow_end <= workflow_start or not math.isclose(
            workflow_latency,
            workflow_end - workflow_start,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise ValueError("workflow latency differs from its timestamps")
        selected = grouped[key(workflow)]
        if len(selected) != (s if arm == "old" else 1):
            raise ValueError("sample fanout differs from workflow HTTP calls")
        starts = []
        ends = []
        hashes = []
        states = []
        decisions = 0
        for sample in selected:
            _same(
                sample.get("logical_schedule_sha256"),
                schedule_hashes[
                    (workflow["concurrency"], workflow["phase"], workflow["round"])
                ],
                "paired logical schedule identity",
            )
            if (
                sample.get("case_id") != workflow["case_id"]
                or sample.get("status_code") != 200
                or sample.get("success") is not True
                or sample.get("error_code") is not None
                or sample.get("request_kind")
                != ("single" if arm == "old" or s == 1 else "batch")
            ):
                raise ValueError("raw sample contradicts its successful workflow")
            hashes.append(_hex(sample.get("request_sha256"), "sample request", HASH))
            _hex(sample.get("response_sha256"), "sample response", HASH)
            _integer(sample.get("request_bytes"), "sample request bytes", minimum=1)
            decisions += _integer(
                sample.get("decisions"), "sample decisions", minimum=1
            )
            start = _number(
                sample.get("started_offset_ms"), "sample start", positive=False
            )
            end = _number(sample.get("completed_offset_ms"), "sample end")
            latency = _number(sample.get("latency_ms"), "sample latency")
            if end <= start or not math.isclose(
                latency, end - start, rel_tol=1e-6, abs_tol=1e-6
            ):
                raise ValueError("sample latency differs from its timestamps")
            starts.append(start)
            ends.append(end)
            states.append(sample.get("state_id"))
            by_cell_phase[(workflow["concurrency"], arm, workflow["phase"])].append(
                sample
            )
        if (
            decisions != workflow["decisions"]
            or not math.isclose(
                min(starts), workflow["started_offset_ms"], abs_tol=1e-6
            )
            or not math.isclose(
                max(ends), workflow["completed_offset_ms"], abs_tol=1e-6
            )
        ):
            raise ValueError("sample decisions or time span differ from workflow")
        expected_old, expected_new = requests[workflow["case_id"]]
        if arm == "old":
            if (
                len(set(states)) != s
                or any(not isinstance(state, str) or not state for state in states)
                or sorted(hashes) != sorted(expected_old)
            ):
                raise ValueError("old sample requests differ from audited case")
        elif states != [None] or hashes != [expected_new]:
            raise ValueError("new sample request differs from audited case")
    for row in raw_rows:
        concurrency = row["concurrency"]
        arms = row["summary"]["arms"]
        for arm in ("old", "new"):
            for phase in ("warmup", "latency", "throughput"):
                phase_samples = by_cell_phase[(concurrency, arm, phase)]
                summary = _mapping(arms[arm].get(phase), arm + " " + phase)
                phase_workflows = [
                    workflow
                    for workflow in workflows
                    if workflow["concurrency"] == concurrency
                    and workflow["arm"] == arm
                    and workflow["phase"] == phase
                ]
                for field, expected in (
                    ("attempted_workflows", len(phase_workflows)),
                    ("successful_workflows", len(phase_workflows)),
                    ("failed_workflows", 0),
                    (
                        "attempted_decisions",
                        sum(workflow["decisions"] for workflow in phase_workflows),
                    ),
                    (
                        "successful_decisions",
                        sum(workflow["decisions"] for workflow in phase_workflows),
                    ),
                ):
                    _same(summary.get(field), expected, arm + " " + phase + " " + field)
                _percentiles(
                    [workflow["latency_ms"] for workflow in phase_workflows],
                    summary.get("workflow_latency"),
                    arm + " " + phase + " workflow latency",
                )
                _same(
                    summary.get("http_attempts"),
                    len(phase_samples),
                    arm + " " + phase + " HTTP attempts",
                )
                _same(
                    summary.get("http_successes"),
                    len(phase_samples),
                    arm + " " + phase + " HTTP successes",
                )
                _same(summary.get("errors"), {}, arm + " " + phase + " errors")
                _percentiles(
                    [sample["latency_ms"] for sample in phase_samples],
                    summary.get("http_latency"),
                    arm + " " + phase + " HTTP latency",
                )
    return samples


def _validate_metrics(
    path: Path,
    raw_rows: list[dict[str, Any]],
    settings: dict[str, Any],
    q: int,
    s: int,
    new_physical_batch_size: int,
    runtime_variant: str,
    previous_snapshots: dict[str, dict[str, dict[str, Any]]],
    hash_snapshots: dict[tuple[str, str], str],
) -> None:
    rows = _jsonl(path, "metrics")
    requested = _mapping(settings.get("metrics_collection"), "metrics policy")
    if requested.get("new") is not True or type(requested.get("old")) is not bool:
        raise ValueError("new metrics must be collected")
    expected = {
        (concurrency, arm, round_number)
        for concurrency in CONCURRENCIES
        for arm in ("old", "new")
        if requested[arm]
        for round_number in range(ROUNDS)
    }
    indexed = {}
    for record in rows:
        key = (record.get("concurrency"), record.get("arm"), record.get("round"))
        if (
            key not in expected
            or key in indexed
            or record.get("question_count") != q
            or record.get("state_count") != s
            or record.get("error_code") is not None
        ):
            raise ValueError("metric capture inventory or status differs")
        indexed[key] = record
    if set(indexed) != expected:
        raise ValueError("metric capture inventory is incomplete")
    for cell in raw_rows:
        concurrency = cell["concurrency"]
        telemetry = _mapping(cell["summary"]["telemetry"]["arms"], "telemetry arms")
        for arm in ("old", "new"):
            summary = _mapping(telemetry.get(arm), arm + " telemetry")
            if not requested[arm]:
                _same(summary.get("status"), "not_requested", arm + " telemetry")
                continue
            totals: Counter[str] = Counter()
            bucket_totals: Counter[str] = Counter()
            graph_totals: Counter[str] = Counter()
            for round_number in range(ROUNDS):
                capture = indexed[(concurrency, arm, round_number)]
                before = _mapping(capture.get("before"), "metric before")
                after = _mapping(capture.get("after"), "metric after")
                delta = _mapping(capture.get("delta"), "metric delta")
                before_counters = _mapping(before.get("counters"), "before counters")
                after_counters = _mapping(after.get("counters"), "after counters")
                delta_counters = _mapping(delta.get("counters"), "delta counters")
                before_buckets = _mapping(
                    before.get("physical_batch_size_buckets"), "before buckets"
                )
                after_buckets = _mapping(
                    after.get("physical_batch_size_buckets"), "after buckets"
                )
                delta_buckets = _mapping(
                    delta.get("physical_batch_size_buckets"), "delta buckets"
                )
                before_graph = _mapping(
                    before.get("graph_events"), "before graph events"
                )
                after_graph = _mapping(after.get("graph_events"), "after graph events")
                delta_graph = _mapping(delta.get("graph_events"), "delta graph events")
                if any(
                    set(events) != GRAPH_EVENTS
                    for events in (before_graph, after_graph, delta_graph)
                ):
                    raise ValueError("graph event inventory differs")
                for snapshot, label in ((before, "before"), (after, "after")):
                    digest = _hex(
                        snapshot.get("response_sha256"), label + " metrics", HASH
                    )
                    canonical = json.dumps(
                        {
                            "counters": snapshot.get("counters"),
                            "physical_batch_size_buckets": snapshot.get(
                                "physical_batch_size_buckets"
                            ),
                            "graph_events": snapshot.get("graph_events"),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    identity = (arm, digest)
                    previous_content = hash_snapshots.setdefault(identity, canonical)
                    if previous_content != canonical:
                        raise ValueError(
                            "identical metrics response hash has different counters"
                        )
                bucket_names = {
                    *(str(bound) for bound in PHYSICAL_BATCH_BUCKETS),
                    "+Inf",
                }
                if (
                    set(before_buckets) != bucket_names
                    or set(after_buckets) != bucket_names
                    or set(delta_buckets) != bucket_names
                ):
                    raise ValueError("metric histogram bucket inventory differs")
                for snapshot, counters_snapshot in (
                    (before_buckets, before_counters),
                    (after_buckets, after_counters),
                ):
                    previous = 0
                    for bucket in (*map(str, PHYSICAL_BATCH_BUCKETS), "+Inf"):
                        count = _counter_integer(
                            snapshot[bucket], "cumulative histogram bucket"
                        )
                        if count < previous:
                            raise ValueError("metric histogram is not cumulative")
                        previous = count
                    _same(
                        previous,
                        _counter_integer(
                            counters_snapshot.get("physical_batches"),
                            "snapshot physical batches",
                        ),
                        "histogram snapshot batch count",
                    )
                previous_delta = 0
                for bucket, observed in delta_buckets.items():
                    delta_value = _counter_integer(observed, "histogram bucket delta")
                    arithmetic = _number(
                        after_buckets[bucket], "histogram after", positive=False
                    ) - _number(
                        before_buckets[bucket], "histogram before", positive=False
                    )
                    if not math.isclose(
                        delta_value, arithmetic, rel_tol=1e-9, abs_tol=1e-9
                    ):
                        raise ValueError("metric histogram differs from snapshots")
                    bucket_totals[bucket] += delta_value
                for bucket in (*map(str, PHYSICAL_BATCH_BUCKETS), "+Inf"):
                    cumulative_delta = _counter_integer(
                        delta_buckets[bucket], "cumulative histogram delta"
                    )
                    if cumulative_delta < previous_delta:
                        raise ValueError("metric histogram delta is not cumulative")
                    previous_delta = cumulative_delta
                for name in (
                    "row_preparation_seconds",
                    "row_preparations",
                    "physical_batches",
                    "physical_batch_rows",
                ):
                    observed = _number(
                        delta_counters.get(name), name + " metric delta", positive=False
                    )
                    arithmetic = _number(
                        after_counters.get(name), name + " metric after", positive=False
                    ) - _number(
                        before_counters.get(name),
                        name + " metric before",
                        positive=False,
                    )
                    if not math.isclose(
                        observed, arithmetic, rel_tol=1e-9, abs_tol=1e-9
                    ):
                        raise ValueError("metric delta differs from snapshots")
                    totals[name] += observed
                for name in GRAPH_EVENTS:
                    before_count = _counter_integer(
                        before_graph[name], "graph event before"
                    )
                    after_count = _counter_integer(
                        after_graph[name], "graph event after"
                    )
                    delta_count = _counter_integer(
                        delta_graph[name], "graph event delta"
                    )
                    if after_count - before_count != delta_count:
                        raise ValueError("graph event delta differs from snapshots")
                    graph_totals[name] += delta_count
                    if (
                        arm == "new"
                        and runtime_variant == "eager"
                        and (before_count or after_count or delta_count)
                    ):
                        raise ValueError("eager candidate emitted graph events")
                if (
                    arm == "new"
                    and runtime_variant == "sol_rocm_graph_b8"
                    and concurrency == 32
                    and (q, s) in HIGH_LOAD_SHAPES
                    and not delta_graph["replay"]
                ):
                    raise ValueError("graph candidate has no timed high-load replay")
                round_batches = _counter_integer(
                    delta_counters.get("physical_batches"),
                    "round physical batches",
                    minimum=1,
                )
                round_rows = _counter_integer(
                    delta_counters.get("physical_batch_rows"),
                    "round physical rows",
                    minimum=1,
                )
                _same(
                    delta_buckets["+Inf"],
                    round_batches,
                    "histogram +Inf physical batch count",
                )
                if arm == "new":
                    round_summary = cell["summary"]["arms"]["new"]["throughput"][
                        "rounds"
                    ][round_number]
                    successful_workflows = _integer(
                        round_summary.get("successful_workflows"),
                        "round successful workflows",
                        minimum=1,
                    )
                    _same(
                        _counter_integer(
                            delta_counters.get("row_preparations"),
                            "round row preparations",
                        ),
                        successful_workflows,
                        "new round row preparation coverage",
                    )
                    _same(
                        round_rows,
                        q * s * successful_workflows,
                        "new round physical row coverage",
                    )
                    if (
                        not round_batches
                        <= round_rows
                        <= round_batches * new_physical_batch_size
                    ):
                        raise ValueError("new round physical batch capacity differs")
                    previous_bound = 0
                    min_rows = 0
                    max_rows = 0
                    previous_count = 0
                    for bound in PHYSICAL_BATCH_BUCKETS:
                        cumulative = _counter_integer(
                            delta_buckets[str(bound)], "histogram bucket delta"
                        )
                        count = cumulative - previous_count
                        if count:
                            if previous_bound >= new_physical_batch_size:
                                raise ValueError(
                                    "histogram exceeds configured batch capacity"
                                )
                            min_rows += count * (previous_bound + 1)
                            max_rows += count * min(bound, new_physical_batch_size)
                        previous_count = cumulative
                        previous_bound = bound
                    if (
                        previous_count != round_batches
                        or not min_rows <= round_rows <= max_rows
                    ):
                        raise ValueError(
                            "histogram contradicts physical row count or capacity"
                        )
            _same(summary.get("status"), "complete", arm + " telemetry status")
            _same(summary.get("rounds"), ROUNDS, arm + " telemetry rounds")
            counters = _mapping(summary.get("counter_deltas"), arm + " counters")
            buckets = _mapping(
                summary.get("physical_batch_size_bucket_deltas"),
                arm + " histogram",
            )
            graph_summary = _mapping(
                summary.get("graph_event_deltas"), arm + " graph events"
            )
            if set(graph_summary) != GRAPH_EVENTS:
                raise ValueError("graph event summary inventory differs")
            for name in GRAPH_EVENTS:
                _same(
                    _counter_integer(graph_summary[name], name + " graph total"),
                    graph_totals[name],
                    name + " graph total",
                )
            _same(set(buckets), set(bucket_totals), arm + " histogram buckets")
            for bucket, value in bucket_totals.items():
                if not math.isclose(
                    _number(buckets[bucket], bucket + " bucket", positive=False),
                    value,
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                ):
                    raise ValueError("metric histogram total differs from captures")
            for name, value in totals.items():
                if not math.isclose(
                    _number(counters.get(name), name + " counter", positive=False),
                    value,
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                ):
                    raise ValueError("metric total differs from raw captures")
            batches = totals["physical_batches"]
            if batches <= 0:
                raise ValueError("no physical batches were observed")
            _same(
                summary.get("normalization_decisions"),
                cell["summary"]["arms"][arm]["throughput"]["successful_decisions"],
                arm + " telemetry normalization",
            )
            decisions = summary["normalization_decisions"]
            for field, numerator in (
                ("row_preparation_seconds_per_decision", "row_preparation_seconds"),
                ("row_preparations_per_decision", "row_preparations"),
                ("physical_batch_rows_per_decision", "physical_batch_rows"),
            ):
                _close(
                    summary.get(field),
                    totals[numerator] / decisions,
                    arm + " " + field,
                )
            _close(
                summary.get("observed_rows_per_physical_batch"),
                totals["physical_batch_rows"] / batches,
                arm + " observed batch size",
            )
    expected_order = [
        (concurrency, arm, round_number)
        for concurrency in CONCURRENCIES
        for round_number in range(ROUNDS)
        for arm in (("old", "new") if round_number % 2 == 0 else ("new", "old"))
        if requested[arm]
    ]
    if [
        (row["concurrency"], row["arm"], row["round"]) for row in rows
    ] != expected_order:
        raise ValueError("metric captures are out of producer order")
    for row in rows:
        arm = row["arm"]
        before = row["before"]
        after = row["after"]
        previous = previous_snapshots.get(arm)
        if previous is not None:
            for name in (
                "row_preparation_seconds",
                "row_preparations",
                "physical_batches",
                "physical_batch_rows",
            ):
                current = _number(
                    before["counters"].get(name),
                    "metric counter before",
                    positive=False,
                )
                earlier = _number(
                    previous["counters"].get(name),
                    "prior metric counter",
                    positive=False,
                )
                if current < earlier and not math.isclose(
                    current, earlier, rel_tol=1e-12, abs_tol=1e-12
                ):
                    raise ValueError("metric counter reset between measured waves")
            for bucket in (*map(str, PHYSICAL_BATCH_BUCKETS), "+Inf"):
                current = _counter_integer(
                    before["physical_batch_size_buckets"].get(bucket),
                    "metric histogram before",
                )
                earlier = _counter_integer(
                    previous["physical_batch_size_buckets"].get(bucket),
                    "prior metric histogram",
                )
                if current < earlier:
                    raise ValueError("metric histogram reset between measured waves")
            for name in GRAPH_EVENTS:
                current = _counter_integer(
                    before["graph_events"].get(name), "graph event before"
                )
                earlier = _counter_integer(
                    previous["graph_events"].get(name), "prior graph event"
                )
                if current < earlier:
                    raise ValueError("graph event counter reset between measured waves")
        previous_snapshots[arm] = after


def _validate_shape(
    shape: dict[str, Any],
    root: Path,
    model: dict[str, Any],
    source_sha: str,
    harness: str,
    new_physical_batch_size: int,
    previous_snapshots: dict[str, dict[str, dict[str, Any]]],
    hash_snapshots: dict[tuple[str, str], str],
) -> None:
    q, s = shape.get("question_count"), shape.get("state_count")
    label = f"{model['model_id']} q{q}/s{s}"
    if (
        shape.get("preflight_status") != "passed"
        or shape.get("formal_audit_status") != "passed"
    ):
        raise ValueError(f"{label} semantic audit did not pass")
    _clean_mismatches(shape.get("preflight_mismatch_counts"), label + " preflight")
    _clean_mismatches(
        shape.get("formal_audit_mismatch_counts"), label + " formal audit"
    )
    if (
        _number(
            shape.get("preflight_max_absolute_probability_delta"),
            label + " probability delta",
            positive=False,
        )
        > MAX_SEMANTIC_PROBABILITY_DELTA
    ):
        raise ValueError(f"{label} semantic probability delta exceeds tolerance")
    _same(shape.get("harness_sha256"), harness, label + " harness")
    _same(shape.get("arrival_policy"), ARRIVAL_POLICY, label + " arrival policy")
    _same(shape.get("benchmark_source_sha"), source_sha, label + " benchmark source")
    _same(shape.get("new_runtime_source_sha"), source_sha, label + " runtime source")
    receipt = _read_json(
        _evidence(
            root,
            shape.get("raw_receipt_path"),
            shape.get("raw_receipt_sha256"),
            label + " receipt",
        )
    )
    preflight = _read_json(
        _evidence(
            root,
            shape.get("raw_preflight_path"),
            shape.get("raw_preflight_sha256"),
            label + " preflight",
        )
    )
    workflows = _evidence(
        root,
        shape.get("raw_workflows_path"),
        shape.get("raw_workflows_sha256"),
        label + " workflows",
    )
    other_evidence = {}
    for name in ("preflight_audit", "audit", "samples", "metrics", "timed_semantic"):
        other_evidence[name] = _evidence(
            root,
            shape.get(f"raw_{name}_path"),
            shape.get(f"raw_{name}_sha256"),
            label + " " + name,
        )
    _same(preflight.get("status"), "passed", label + " raw preflight")
    _same(
        preflight.get("mismatch_counts"),
        shape["preflight_mismatch_counts"],
        label + " raw preflight mismatches",
    )
    _same(receipt.get("schema_version"), RAW_SCHEMA, label + " raw schema")
    _same(receipt.get("model"), model["model_id"], label + " raw model")
    _same(receipt.get("status"), "measured", label + " raw status")
    _same(receipt.get("source_commit"), source_sha, label + " raw source")
    _same(receipt.get("harness_sha256"), harness, label + " raw harness")
    _same(receipt.get("failed_workflows"), 0, label + " failed workflows")
    _same(receipt.get("failed_metrics_shapes"), 0, label + " failed metrics shapes")
    audit = _mapping(receipt.get("audit"), label + " raw audit")
    if audit.get("status") != "passed" or audit.get("comparison_eligible") is not True:
        raise ValueError(f"{label} raw formal audit is ineligible")
    _same(
        audit.get("mismatch_counts"),
        shape["formal_audit_mismatch_counts"],
        label + " raw formal mismatches",
    )
    settings = _mapping(receipt.get("settings"), label + " raw settings")
    case_ids = _list(receipt.get("case_ids"), label + " case IDs")
    if (
        len(case_ids) != 4
        or len(set(case_ids)) != len(case_ids)
        or any(not isinstance(case_id, str) or not case_id for case_id in case_ids)
    ):
        raise ValueError(label + " case ID inventory is invalid")
    for field, expected in (
        ("question_counts", [q]),
        ("state_counts", [s]),
        ("concurrencies", list(CONCURRENCIES)),
        ("throughput_rounds", ROUNDS),
        ("parity_policy", "require"),
        ("arrival_policy", ARRIVAL_POLICY),
        ("variants_per_shape", 4),
        ("seed", 17),
        ("warmup_workflows_per_arm", 2),
        ("latency_workflows_per_arm", 16),
        ("timed_semantic_evidence", True),
    ):
        _same(settings.get(field), expected, label + " " + field)
    if (
        _integer(
            settings.get("throughput_workflows_per_round_per_arm"), label + " workflows"
        )
        < MIN_WORKFLOWS_PER_ROUND
    ):
        raise ValueError(f"{label} has too few throughput workflows")
    for arm in ("old", "new"):
        provenance = _mapping(receipt.get(arm), label + " " + arm)
        _same(
            provenance.get("model_revision"),
            model["same_old_new_revision"],
            label + " model revision",
        )
        _same(
            provenance.get("declared_physical_batch_size"),
            (
                model["old_physical_batch_size"]
                if arm == "old"
                else new_physical_batch_size
            ),
            label + " declared physical batch",
        )
        if arm == "old":
            _same(
                provenance.get("source_ref"),
                "sha256:" + model["old_core_source_sha256"],
                label + " old core source",
            )
        else:
            _same(provenance.get("source_ref"), source_sha, label + " new source")
    old_provenance = receipt["old"]
    new_provenance = receipt["new"]
    adapter = _mapping(receipt.get("adapter"), label + " adapter")
    old_model_id = adapter.get("old_model_id")
    if not isinstance(old_model_id, str) or not old_model_id:
        raise ValueError(label + " old model adapter ID is invalid")
    if (
        adapter.get("envelope_transform")
        not in (
            "none",
            "old_max_probability_to_decision_v1_for_validation",
        )
        or adapter.get("applied_outside_timed_interval") is not True
    ):
        raise ValueError(label + " response adapter is invalid")
    same_wire_bytes = old_model_id == model["model_id"]
    expected_adapter = (
        ("legacy_preview" if same_wire_bytes else "legacy_preview_and_model_id")
        if adapter.get("envelope_transform") != "none"
        else ("none" if same_wire_bytes else "model_id_only")
    )
    _same(adapter.get("kind"), expected_adapter, label + " adapter kind")
    _same(
        old_provenance.get("hardware"),
        new_provenance.get("hardware"),
        label + " hardware",
    )
    _same(old_provenance.get("network_scope"), "loopback", label + " old network")
    _same(new_provenance.get("network_scope"), "loopback", label + " new network")
    rows = _list(receipt.get("shapes"), label + " raw cells")
    cells = _list(shape.get("cells"), label + " cells")
    if len(rows) != len(CONCURRENCIES) or len(cells) != len(CONCURRENCIES):
        raise ValueError(f"{label} requires three concurrency cells")
    for expected, cell, row in zip(CONCURRENCIES, cells, rows, strict=True):
        _same(row.get("case_ids"), case_ids, label + " shape case IDs")
        _same(cell.get("concurrency"), expected, label + " concurrency")
        _validate_cell(
            _mapping(cell, label + " cell"),
            _mapping(row, label + " raw cell"),
            q,
            s,
            same_wire_bytes=same_wire_bytes,
            new_physical_batch_size=new_physical_batch_size,
        )
    formal_requests = _audit_rows(
        other_evidence["audit"], audit, case_ids, q, s, label + " formal audit"
    )
    preflight_requests = _audit_rows(
        other_evidence["preflight_audit"],
        preflight,
        case_ids,
        q,
        s,
        label + " preflight audit",
    )
    _same(preflight_requests, formal_requests, label + " audited request identities")
    _same(
        preflight["absolute_probability_delta"]["max"],
        shape["preflight_max_absolute_probability_delta"],
        label + " preflight probability max",
    )
    workflows_rows = _validate_workflows(workflows, rows, q, s, case_ids, settings)
    sample_rows = _validate_samples(
        other_evidence["samples"], workflows_rows, rows, formal_requests, q, s
    )
    validate_timed_semantics(
        other_evidence["timed_semantic"],
        sample_rows,
        _jsonl(other_evidence["audit"], label + " formal audit"),
        q=q,
        s=s,
        model_id=model["model_id"],
        rounds=ROUNDS,
    )
    _validate_metrics(
        other_evidence["metrics"],
        rows,
        settings,
        q,
        s,
        new_physical_batch_size,
        model["new_runtime_variant"],
        previous_snapshots,
        hash_snapshots,
    )


def validate_report(
    path: Path,
    *,
    source_sha: str,
    qualification: dict[str, Any],
    candidate_ref: str,
    run_id: str | None = None,
    run_attempt: str | None = None,
) -> dict[str, Any]:
    """Validate an exact-source report and return a compact verified summary."""
    _hex(source_sha, "source SHA", SHA)
    report = _read_json(path)
    _same(report.get("schema_version"), SCHEMA, "performance report schema")
    _same(
        report.get("scope"),
        "synthetic same-revision Decision HTTP performance, not task-quality evaluation",
        "performance claim scope",
    )
    _same(report.get("source_sha"), source_sha, "performance source")
    _hex(candidate_ref, "candidate reference", IMAGE_REF)
    _same(qualification.get("source_sha"), source_sha, "ROCm qualification source")
    _same(qualification.get("candidate_ref"), candidate_ref, "ROCm candidate")
    _same(report.get("candidate_ref"), candidate_ref, "performance candidate")
    qualified_models = _list(qualification.get("models"), "ROCm qualified models")
    if (
        len(qualified_models) != len(MODEL_IDS)
        or {row.get("id") for row in qualified_models if isinstance(row, dict)}
        != MODEL_IDS
    ):
        raise ValueError("ROCm qualification model inventory differs")
    qualified = {row["id"]: row for row in qualified_models}
    if run_id is not None:
        _same(report.get("run_id"), run_id, "protected run ID")
    if run_attempt is not None:
        _same(report.get("run_attempt"), run_attempt, "protected run attempt")
    _same(
        report.get("new_runtime_image_source_sha"), source_sha, "candidate image source"
    )
    _same(report.get("source_image_match"), True, "source and image match")
    harness = _hex(report.get("harness_sha256"), "harness", HASH)
    environment = _mapping(report.get("environment"), "environment")
    _same(
        environment.get("physical_batch_policy"),
        "per-model launch size; occupancy from per-round metrics",
        "physical batch policy",
    )
    _same(environment.get("throughput_rounds_per_cell"), ROUNDS, "round policy")
    _same(environment.get("concurrencies"), list(CONCURRENCIES), "concurrency policy")
    _same(
        environment.get("workload_shapes"),
        [{"questions": q, "states": s} for q, s in SHAPES],
        "shape policy",
    )
    _same(environment.get("arrival_policy"), ARRIVAL_POLICY, "arrival policy")
    _same(
        environment.get("gpu_exclusivity"),
        "dedicated_gpu_no_unrelated_compute; externally attested by protected runner",
        "GPU exclusivity scope",
    )
    if environment.get("gpu_clock_policy") not in (
        "unobserved",
        "protected_fixed",
        "default_dynamic",
    ):
        raise ValueError("GPU clock policy is invalid")
    _same(
        environment.get("service_residency"),
        "both service containers running during alternating waves; HBM residency unmeasured",
        "service residency scope",
    )
    _same(environment.get("new_max_concurrency"), 4, "new scheduler concurrency")
    _same(environment.get("new_max_queue"), 32, "new scheduler queue")
    if "AMD Instinct" not in str(
        environment.get("hardware", "")
    ) or "loopback" not in str(environment.get("network", "")):
        raise ValueError("hardware or network comparison scope is invalid")
    models = _list(report.get("models"), "models")
    if (
        len(models) != len(MODEL_IDS)
        or {row.get("model_id") for row in models if isinstance(row, dict)} != MODEL_IDS
    ):
        raise ValueError(
            "performance report requires exactly six distinct Decision models"
        )
    new_images: set[str] = set()
    all_high_load_ratios: list[float] = []
    summary: dict[str, Any] = {"source_sha": source_sha, "models": {}}
    for entry in models:
        model = _mapping(entry, "model")
        model_id = model["model_id"]
        _hex(model.get("same_old_new_revision"), model_id + " revision", SHA)
        for field in (
            "same_old_new_artifact_content_id",
            "artifact_metadata_sha256",
            "artifact_manifest_sha256",
            "old_core_source_sha256",
        ):
            _hex(model.get(field), model_id + " " + field, HASH)
        _same(
            model["same_old_new_revision"],
            qualified[model_id].get("revision"),
            model_id + " qualified revision",
        )
        _same(
            "sha256:" + model["same_old_new_artifact_content_id"],
            qualified[model_id].get("artifact_content_id"),
            model_id + " qualified artifact",
        )
        _same(
            model.get("old_artifact_layout"),
            "full_snapshot_selected_data_v1",
            model_id + " old artifact layout",
        )
        old_snapshot_digest = _hex(
            model.get("old_full_snapshot_sha256"),
            model_id + " declared old full snapshot",
            HASH,
        )
        snapshot_verifications = _list(
            model.get("old_snapshot_verifications"),
            model_id + " old snapshot verifications",
        )
        if len(snapshot_verifications) != 2 or any(
            not isinstance(value, dict)
            or set(value) != {"full_snapshot_sha256", "selected_data_content_id"}
            for value in snapshot_verifications
        ):
            raise ValueError(model_id + " needs two full-snapshot verifications")
        for verification in snapshot_verifications:
            _hex(
                verification["full_snapshot_sha256"],
                model_id + " old full snapshot",
                HASH,
            )
            _same(
                verification["full_snapshot_sha256"],
                old_snapshot_digest,
                model_id + " declared old full snapshot",
            )
            _same(
                _hex(
                    verification["selected_data_content_id"],
                    model_id + " old selected data",
                    HASH,
                ),
                model["same_old_new_artifact_content_id"],
                model_id + " old selected-data content identity",
            )
        if snapshot_verifications[0] != snapshot_verifications[1]:
            raise ValueError(model_id + " old full snapshot changed during measurement")
        if model.get("old_core_source_kind") == "mounted_adapter" and all(
            field in model
            for field in (
                "old_adapter_source_sha256",
                "old_source_declaration",
                "old_attestation_path",
                "old_attestation_sha256",
            )
        ):
            declaration = _mapping(
                model["old_source_declaration"], model_id + " old source declaration"
            )
            if set(declaration) != {
                "adapter_path_sha256",
                "imported_core_path_sha256",
                "imported_module_sha256",
                "artifact_mount_path_sha256",
            }:
                raise ValueError(model_id + " old source declaration is invalid")
            for field, digest in declaration.items():
                _hex(digest, model_id + " old " + field, HASH)
            adapter_sha = _hex(
                model.get("old_adapter_source_sha256"),
                model_id + " old adapter source",
                HASH,
            )
            attestation_path = _evidence(
                path.parent,
                model.get("old_attestation_path"),
                model.get("old_attestation_sha256"),
                model_id + " old live attestation",
            )
            attestation = _read_json(attestation_path)
            _same(
                attestation.get("schema_version"),
                "decision-old-baseline-attestation-v1",
                model_id + " old attestation schema",
            )
            observations = _list(
                attestation.get("observations"), model_id + " old observations"
            )
            if len(observations) != 2 or any(
                not isinstance(row, dict) for row in observations
            ):
                raise ValueError(model_id + " needs pre/post old process attestations")
            if observations[0] == observations[1]:
                raise ValueError(model_id + " old process attestations were replayed")
            for observation in observations:
                if set(observation) != {
                    "schema_version",
                    "challenge",
                    "pid",
                    "process_start_ticks",
                    "adapter_path_sha256",
                    "adapter_sha256",
                    "imported_module_sha256",
                    "imported_core_path_sha256",
                    "core_mount_sha256",
                    "loaded_artifact_root_sha256",
                    "loaded_artifact_content_id",
                    "model_id",
                    "revision",
                }:
                    raise ValueError(
                        model_id + " old process attestation fields differ"
                    )
                if (
                    _integer(observation.get("pid"), model_id + " old pid", minimum=1)
                    != 1
                ):
                    raise ValueError(model_id + " old attested process is not PID 1")
                for key, expected in (
                    ("schema_version", "decision-old-baseline-attestation-v1"),
                    ("pid", 1),
                    ("adapter_sha256", adapter_sha),
                    ("core_mount_sha256", model["old_core_source_sha256"]),
                    (
                        "loaded_artifact_content_id",
                        model["same_old_new_artifact_content_id"],
                    ),
                    ("model_id", model_id),
                    ("revision", model["same_old_new_revision"]),
                    ("adapter_path_sha256", declaration["adapter_path_sha256"]),
                    ("imported_module_sha256", declaration["imported_module_sha256"]),
                    (
                        "imported_core_path_sha256",
                        declaration["imported_core_path_sha256"],
                    ),
                    (
                        "loaded_artifact_root_sha256",
                        declaration["artifact_mount_path_sha256"],
                    ),
                ):
                    _same(observation.get(key), expected, model_id + " old live " + key)
                _hex(observation.get("challenge"), model_id + " old challenge", HASH)
                _integer(
                    observation.get("process_start_ticks"),
                    model_id + " old start ticks",
                    minimum=1,
                )
            if observations[0]["challenge"] == observations[1]["challenge"]:
                raise ValueError(model_id + " old process challenge was reused")
            if (
                observations[0]["process_start_ticks"]
                != observations[1]["process_start_ticks"]
            ):
                raise ValueError(model_id + " old process changed during measurement")
        else:
            raise ValueError(model_id + " lacks mandatory live old process proof")
        if not model.get("old_arm_overlay"):
            raise ValueError(model_id + " old arm overlay is missing")
        old_batch = _integer(
            model.get("old_physical_batch_size"),
            model_id + " old physical batch",
            minimum=1,
        )
        new_batch = _integer(
            model.get("new_physical_batch_size"),
            model_id + " new physical batch",
            minimum=1,
        )
        if old_batch > 4096 or new_batch > 4096:
            raise ValueError(model_id + " physical batch exceeds runtime capacity")
        if new_batch > PHYSICAL_BATCH:
            raise ValueError(
                model_id
                + " larger physical batch requires independent tuning qualification evidence"
            )
        variant = model.get("new_runtime_variant")
        if (
            not isinstance(variant, str)
            or variant not in RUNTIME_VARIANTS
            or (
                variant == "sol_rocm_graph_b8"
                and (model_id != SOL_GRAPH_MODEL_ID or new_batch != 8)
            )
        ):
            raise ValueError(model_id + " candidate runtime variant is invalid")
        _same(
            model.get("new_scheduler"),
            {"max_concurrency": 4, "max_queue": 32},
            model_id + " new scheduler policy",
        )
        old_image = _hex(model.get("old_image_id"), model_id + " old image", IMAGE_ID)
        new_image = _hex(model.get("new_image_id"), model_id + " new image", IMAGE_ID)
        if old_image == new_image:
            raise ValueError(model_id + " old and new image identities are identical")
        new_images.add(new_image)
        shapes = _list(model.get("shapes"), model_id + " shapes")
        if len(shapes) != len(SHAPES):
            raise ValueError(model_id + " requires all three workload shapes")
        indexed = {
            (shape.get("question_count"), shape.get("state_count")): shape
            for shape in shapes
            if isinstance(shape, dict)
        }
        if set(indexed) != set(SHAPES):
            raise ValueError(model_id + " has missing or duplicate workload shapes")
        previous_snapshots: dict[str, dict[str, dict[str, Any]]] = {}
        hash_snapshots: dict[tuple[str, str], str] = {}
        for q, s in SHAPES:
            _validate_shape(
                _mapping(indexed[q, s], model_id + " shape"),
                path.parent,
                model,
                source_sha,
                harness,
                new_batch,
                previous_snapshots,
                hash_snapshots,
            )
        # The arms perform the same logical work, but old fanout and new batch
        # have different HTTP submissions and no shared scheduled-arrival trace.
        # Workflow latency is diagnostic, never an alternative release win.
        high_load_shape_ratios = {
            f"q{q}_s{s}_c{cell['concurrency']}": cell[
                "new_over_old_decisions_per_second"
            ]
            for q, s in SHAPES
            for cell in indexed[q, s]["cells"]
            if cell["concurrency"] >= HIGH_LOAD_MIN_CONCURRENCY
        }
        for cell_name, ratio in high_load_shape_ratios.items():
            if ratio < MIN_HIGH_LOAD_THROUGHPUT_RATIO:
                raise ValueError(
                    f"{model_id} has a high-load throughput regression at "
                    f"{cell_name} against its selected historical snapshot"
                )
        model_high_load_geomean = _geometric_mean(list(high_load_shape_ratios.values()))
        for q, s in HIGH_LOAD_SHAPES:
            if (
                indexed[q, s]["cells"][-1]["new_observed_rows_per_physical_batch"]
                <= 1.0
            ):
                raise ValueError(
                    model_id + " did not demonstrate physical batching under load"
                )
        all_high_load_ratios.extend(high_load_shape_ratios.values())
        summary["models"][model_id] = {
            "new_runtime_variant": variant,
            "high_load_shape_ratios": high_load_shape_ratios,
            "high_load_throughput_geomean": model_high_load_geomean,
            "high_load_graph_replays": {
                f"q{q}_s{s}_c32": indexed[q, s]["cells"][-1]["new_graph_replays"]
                for q, s in HIGH_LOAD_SHAPES
            },
        }
    if len(new_images) != 1:
        raise ValueError("six models must use one immutable new runtime image")
    summary["high_load_throughput_geomean"] = _geometric_mean(all_high_load_ratios)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--report", type=Path, required=True)
    validate.add_argument("--source-sha", required=True)
    validate.add_argument("--run-id")
    validate.add_argument("--run-attempt")
    validate.add_argument("--qualification-receipt", type=Path, required=True)
    validate.add_argument("--owner", required=True)
    validate.add_argument("--candidate-ref", required=True)
    args = parser.parse_args()
    try:
        qualification = validate_receipt(
            args.qualification_receipt, owner=args.owner, revision=args.source_sha
        )
        summary = validate_report(
            args.report,
            source_sha=args.source_sha,
            qualification=qualification,
            candidate_ref=args.candidate_ref,
            run_id=args.run_id,
            run_attempt=args.run_attempt,
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"Decision performance qualification failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "qualified", **summary}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
