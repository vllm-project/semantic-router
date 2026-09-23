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
from pathlib import Path
from typing import Any

from decision_rocm_promotion import MODEL_IDS

SCHEMA = "decision-paired-release-v1"
RAW_SCHEMA = "decision-semantic-workload-v2"
SHAPES = ((32, 1), (8, 8), (32, 32))
CONCURRENCIES = (1, 8, 32)
ROUNDS = 3
PHYSICAL_BATCH = 8
MIN_WORKFLOWS_PER_ROUND = 32
MIN_HIGH_LOAD_THROUGHPUT_GAIN = 1.20
MIN_HIGH_LOAD_LATENCY_GAIN = 1.10
MIN_ACCEPTABLE_HIGH_LOAD_THROUGHPUT_RATIO = 0.80
MAX_SEMANTIC_PROBABILITY_DELTA = 0.010000001
HIGH_LOAD_MIN_CONCURRENCY = 8
SHA = re.compile(r"[0-9a-f]{40}\Z")
HASH = re.compile(r"[0-9a-f]{64}\Z")
IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")
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
        or name.suffix not in (".json", ".jsonl")
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
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"{label} content changed")
    return path


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


def _validate_cell(cell: dict[str, Any], raw: dict[str, Any], q: int, s: int) -> None:
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
        else "single_request_model_id_adapter_workflow"
    )
    if comparison.get("type") != expected_protocol:
        raise ValueError("raw comparison protocol is mislabeled")
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
    if rows < batches or rows > batches * PHYSICAL_BATCH:
        raise ValueError("observed physical batch rows exceed declared batch capacity")
    observed = _close(
        cell.get("new_observed_rows_per_physical_batch"),
        measured.get("observed_rows_per_physical_batch"),
        "observed physical batch size",
    )
    _close(observed, rows / batches, "physical batch arithmetic")


def _validate_workflows(
    path: Path, raw_rows: list[dict[str, Any]], q: int, s: int
) -> None:
    """Verify the three alternating, nonoverlapping old/new throughput waves."""
    workflows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(
            line, object_pairs_hook=_unique_pairs, parse_constant=_reject_nonfinite
        )
        workflows.append(_mapping(row, "workflow record"))
    expected_count = 0
    for raw in raw_rows:
        concurrency = raw["concurrency"]
        summary = raw["summary"]
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
        last_end = None
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


def _validate_shape(
    shape: dict[str, Any],
    root: Path,
    model: dict[str, Any],
    source_sha: str,
    harness: str,
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
    for field, expected in (
        ("question_counts", [q]),
        ("state_counts", [s]),
        ("concurrencies", list(CONCURRENCIES)),
        ("throughput_rounds", ROUNDS),
        ("parity_policy", "require"),
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
            PHYSICAL_BATCH,
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
        _same(cell.get("concurrency"), expected, label + " concurrency")
        _validate_cell(
            _mapping(cell, label + " cell"), _mapping(row, label + " raw cell"), q, s
        )
    _validate_workflows(workflows, rows, q, s)


def validate_report(path: Path, *, source_sha: str) -> dict[str, Any]:
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
    _same(
        report.get("new_runtime_image_source_sha"), source_sha, "candidate image source"
    )
    _same(report.get("source_image_match"), True, "source and image match")
    harness = _hex(report.get("harness_sha256"), "harness", HASH)
    environment = _mapping(report.get("environment"), "environment")
    _same(
        environment.get("physical_batch_size"), PHYSICAL_BATCH, "physical batch policy"
    )
    _same(environment.get("throughput_rounds_per_cell"), ROUNDS, "round policy")
    _same(environment.get("concurrencies"), list(CONCURRENCIES), "concurrency policy")
    _same(
        environment.get("workload_shapes"),
        [{"questions": q, "states": s} for q, s in SHAPES],
        "shape policy",
    )
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
        if not model.get("old_arm_overlay"):
            raise ValueError(model_id + " old arm overlay is missing")
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
        for q, s in SHAPES:
            _validate_shape(
                _mapping(indexed[q, s], model_id + " shape"),
                path.parent,
                model,
                source_sha,
                harness,
            )
        high_load = [indexed[8, 8]["cells"][-1], indexed[32, 32]["cells"][-1]]
        high_throughput = max(
            cell["new_over_old_decisions_per_second"] for cell in high_load
        )
        old_p95 = indexed[32, 1]["cells"][-1]["old"]["throughput_window_workflow_ms"][
            "p95_ms"
        ]
        new_p95 = indexed[32, 1]["cells"][-1]["new"]["throughput_window_workflow_ms"][
            "p95_ms"
        ]
        high_latency = old_p95 / new_p95
        if (
            high_throughput < MIN_HIGH_LOAD_THROUGHPUT_GAIN
            and high_latency < MIN_HIGH_LOAD_LATENCY_GAIN
        ):
            raise ValueError(model_id + " lacks a material high-load performance gain")
        for shape in shapes:
            for cell in shape["cells"]:
                if (
                    cell["concurrency"] >= HIGH_LOAD_MIN_CONCURRENCY
                    and cell["new_over_old_decisions_per_second"]
                    < MIN_ACCEPTABLE_HIGH_LOAD_THROUGHPUT_RATIO
                ):
                    raise ValueError(
                        model_id + " has a material high-load throughput regression"
                    )
        for q, s in ((8, 8), (32, 32)):
            if (
                indexed[q, s]["cells"][-1]["new_observed_rows_per_physical_batch"]
                <= 1.0
            ):
                raise ValueError(
                    model_id + " did not demonstrate physical batching under load"
                )
        summary["models"][model_id] = {
            "high_load_throughput_gain": high_throughput,
            "high_load_p95_latency_gain": high_latency,
        }
    if len(new_images) != 1:
        raise ValueError("six models must use one immutable new runtime image")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--report", type=Path, required=True)
    validate.add_argument("--source-sha", required=True)
    args = parser.parse_args()
    try:
        summary = validate_report(args.report, source_sha=args.source_sha)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"Decision performance qualification failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "qualified", **summary}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
