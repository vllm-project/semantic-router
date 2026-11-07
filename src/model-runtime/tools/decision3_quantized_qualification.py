#!/usr/bin/env python3
"""Finalize and validate Decision 3 quantized-package qualification records.

The record references the immutable identities already carried by each package's
``MODEL_MANIFEST.json``. It does not inspect weights, choose a package, or make a
runtime policy decision.
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

from vllm_srun.families.decision3.package import MANIFEST_SCHEMA

FORMAT = "decision3-quantized-qualification/1"
SCHEMES = {"float8", "int4"}
METHODS = {
    "max-per-channel": ("float8", True),
    "awq": ("int4", False),
    "gptq": ("int4", True),
}
DEVICES = {"cuda", "rocm"}
QUESTION_KINDS = {"choice", "noul", "score"}
CONCURRENCIES = {"1", "4", "16"}
INVALID_RECORD = 2
HEX_40 = re.compile(r"[0-9a-f]{40}\Z")
HEX_64 = re.compile(r"[0-9a-f]{64}\Z")
SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")


class QualificationError(ValueError):
    """The record is incomplete, inconsistent, or refers to different bytes."""


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise QualificationError(f"{path} must be an object")
    return value


def _exact_keys(
    value: dict[str, Any],
    path: str,
    required: set[str],
    optional: set[str] = frozenset(),
) -> None:
    missing = sorted(required - set(value))
    extra = sorted(set(value) - required - optional)
    if missing or extra:
        raise QualificationError(
            f"{path} keys differ (missing {missing}, extra {extra})"
        )


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise QualificationError(f"{path} must be a non-empty string")
    return value


def _integer(value: Any, path: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise QualificationError(f"{path} must be an integer >= {minimum}")
    return value


def _number(value: Any, path: str, minimum: float = 0.0) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
        raise QualificationError(f"{path} must be a finite number >= {minimum}")
    return float(value)


def _ratio(value: Any, path: str) -> float:
    result = _number(value, path)
    if result > 1:
        raise QualificationError(f"{path} must be <= 1")
    return result


def _revision(value: Any, path: str) -> str:
    result = _text(value, path)
    if not HEX_40.fullmatch(result):
        raise QualificationError(f"{path} must be a full 40-character commit revision")
    return result


def _hex_digest(value: Any, path: str) -> str:
    result = _text(value, path)
    if not HEX_64.fullmatch(result):
        raise QualificationError(f"{path} must be a 64-character lowercase SHA-256")
    return result


def _dataset(value: Any, path: str) -> dict[str, Any]:
    dataset = _mapping(value, path)
    _exact_keys(dataset, path, {"repo", "revision", "split", "rows", "digest"})
    _text(dataset["repo"], f"{path}.repo")
    _revision(dataset["revision"], f"{path}.revision")
    _text(dataset["split"], f"{path}.split")
    _integer(dataset["rows"], f"{path}.rows", 1)
    digest = _text(dataset["digest"], f"{path}.digest")
    if not SHA256.fullmatch(digest):
        raise QualificationError(f"{path}.digest must be sha256:<64 lowercase hex>")
    return dataset


def _base_splits(dataset: dict[str, Any]) -> set[str]:
    """Upstream splits before optional ``[start:stop]`` row selections."""
    return {part.partition("[")[0] for part in dataset["split"].split("+")}


def _package_ref(value: Any, path: str) -> dict[str, Any]:
    ref = _mapping(value, path)
    _exact_keys(
        ref,
        path,
        {"model_sha256", "manifest_sha256"},
        {"repo", "revision"},
    )
    _hex_digest(ref["model_sha256"], f"{path}.model_sha256")
    _hex_digest(ref["manifest_sha256"], f"{path}.manifest_sha256")
    if ("repo" in ref) != ("revision" in ref):
        raise QualificationError(
            f"{path}.repo and {path}.revision must appear together"
        )
    if "repo" in ref:
        _text(ref["repo"], f"{path}.repo")
        _revision(ref["revision"], f"{path}.revision")
    return ref


def _validate_quality(value: Any, rows: int) -> None:
    quality = _mapping(value, "record.evaluation.quality")
    _exact_keys(
        quality,
        "record.evaluation.quality",
        {"baseline_correct", "candidate_correct", "changed_answers", "by_kind"},
    )
    totals = {
        "rows": rows,
        "baseline_correct": _integer(
            quality["baseline_correct"], "record.evaluation.quality.baseline_correct"
        ),
        "candidate_correct": _integer(
            quality["candidate_correct"], "record.evaluation.quality.candidate_correct"
        ),
        "changed_answers": _integer(
            quality["changed_answers"], "record.evaluation.quality.changed_answers"
        ),
    }
    if max(totals.values()) > rows:
        raise QualificationError("quality counts cannot exceed evaluated rows")
    if (
        abs(totals["baseline_correct"] - totals["candidate_correct"])
        > totals["changed_answers"]
    ):
        raise QualificationError("correct-answer delta cannot exceed changed_answers")

    by_kind = _mapping(quality["by_kind"], "record.evaluation.quality.by_kind")
    _exact_keys(by_kind, "record.evaluation.quality.by_kind", QUESTION_KINDS)
    observed = dict.fromkeys(totals, 0)
    for kind in sorted(QUESTION_KINDS):
        path = f"record.evaluation.quality.by_kind.{kind}"
        counts = _mapping(by_kind[kind], path)
        _exact_keys(counts, path, set(totals))
        kind_rows = _integer(counts["rows"], f"{path}.rows", 1)
        for name in totals:
            count = _integer(counts[name], f"{path}.{name}")
            if name != "rows" and count > kind_rows:
                raise QualificationError(f"{kind} {name} cannot exceed its rows")
            observed[name] += count
        if (
            abs(counts["baseline_correct"] - counts["candidate_correct"])
            > counts["changed_answers"]
        ):
            raise QualificationError(
                f"{kind} correct-answer delta cannot exceed changed_answers"
            )
        if kind == "choice":
            max_changed = min(
                kind_rows,
                2 * kind_rows
                - counts["baseline_correct"]
                - counts["candidate_correct"],
            )
            if counts["changed_answers"] > max_changed:
                raise QualificationError(
                    "choice changed_answers is mathematically impossible"
                )
    if observed != totals:
        raise QualificationError("quality by_kind counts must sum to aggregate counts")


def _validate_reliability(value: Any) -> None:
    path = "record.evaluation.reliability"
    reliability = _mapping(value, path)
    fields = {
        "baseline_errors",
        "candidate_errors",
        "baseline_nan_results",
        "candidate_nan_results",
        "baseline_oom_events",
        "candidate_oom_events",
        "baseline_missing_results",
        "candidate_missing_results",
    }
    _exact_keys(reliability, path, fields)
    for name in fields:
        _integer(reliability[name], f"{path}.{name}")


def _validate_memory(value: Any) -> None:
    path = "record.evaluation.memory"
    memory = _mapping(value, path)
    fields = {
        "baseline_package_bytes",
        "candidate_package_bytes",
        "baseline_loaded_device_bytes",
        "candidate_loaded_device_bytes",
        "baseline_peak_bytes",
        "candidate_peak_bytes",
    }
    _exact_keys(memory, path, fields)
    for name in fields:
        _integer(memory[name], f"{path}.{name}", 1)
    for side in ("baseline", "candidate"):
        if memory[f"{side}_peak_bytes"] < memory[f"{side}_loaded_device_bytes"]:
            raise QualificationError(
                f"{side} peak memory cannot be below loaded memory"
            )


def _validate_performance(value: Any, rows: int) -> None:
    path = "record.evaluation.performance"
    performance = _mapping(value, path)
    _exact_keys(
        performance,
        path,
        {
            "requests",
            "baseline_load_seconds",
            "candidate_load_seconds",
            "latency_ms",
            "throughput_rps",
        },
    )
    if _integer(performance["requests"], f"{path}.requests", 1) > rows:
        raise QualificationError("performance requests cannot exceed evaluated rows")
    for name in ("baseline_load_seconds", "candidate_load_seconds"):
        _number(performance[name], f"{path}.{name}", 0.000001)

    latency_path = f"{path}.latency_ms"
    latency = _mapping(performance["latency_ms"], latency_path)
    _exact_keys(latency, latency_path, {"baseline", "candidate"})
    for side in ("baseline", "candidate"):
        side_path = f"{latency_path}.{side}"
        values = _mapping(latency[side], side_path)
        _exact_keys(values, side_path, {"median", "p95"})
        median = _number(values["median"], f"{side_path}.median", 0.000001)
        p95 = _number(values["p95"], f"{side_path}.p95", 0.000001)
        if p95 < median:
            raise QualificationError(f"{side} p95 latency cannot be below median")

    throughput_path = f"{path}.throughput_rps"
    throughput = _mapping(performance["throughput_rps"], throughput_path)
    _exact_keys(throughput, throughput_path, CONCURRENCIES)
    for concurrency in sorted(CONCURRENCIES):
        values_path = f"{throughput_path}.{concurrency}"
        values = _mapping(throughput[concurrency], values_path)
        _exact_keys(values, values_path, {"baseline", "candidate"})
        for side in ("baseline", "candidate"):
            _number(values[side], f"{values_path}.{side}")


def _validate_core(record: dict[str, Any]) -> None:
    _exact_keys(
        record,
        "record",
        {
            "format",
            "policy",
            "baseline",
            "candidate",
            "quantization",
            "evaluation",
            "gates",
        },
        {"outcome"},
    )
    if record["format"] != FORMAT:
        raise QualificationError(f"record.format must be {FORMAT!r}")

    policy = _mapping(record["policy"], "record.policy")
    _exact_keys(policy, "record.policy", {"issue"})
    issue = _text(policy["issue"], "record.policy.issue")
    if not re.fullmatch(r"https://github\.com/[^/]+/[^/]+/issues/[1-9][0-9]*", issue):
        raise QualificationError("record.policy.issue must be a GitHub issue URL")

    baseline = _package_ref(record["baseline"], "record.baseline")
    if "repo" not in baseline:
        raise QualificationError("record.baseline must pin a repository and revision")
    candidate = _package_ref(record["candidate"], "record.candidate")
    if baseline["model_sha256"] == candidate["model_sha256"]:
        raise QualificationError(
            "candidate must have a different model identity from baseline"
        )

    quantization = _mapping(record["quantization"], "record.quantization")
    _exact_keys(
        quantization,
        "record.quantization",
        {
            "source",
            "scheme",
            "method",
            "group_size",
            "symmetric",
            "calibration",
            "calibration_evaluation_overlap_rows",
        },
    )
    source = _package_ref(quantization["source"], "record.quantization.source")
    if source != baseline:
        raise QualificationError(
            "record.quantization.source must equal record.baseline"
        )
    scheme = _text(quantization["scheme"], "record.quantization.scheme")
    method = _text(quantization["method"], "record.quantization.method")
    if scheme not in SCHEMES:
        raise QualificationError(
            f"record.quantization.scheme must be one of {sorted(SCHEMES)}"
        )
    group_size = quantization["group_size"]
    calibration = quantization["calibration"]
    overlap_rows = quantization["calibration_evaluation_overlap_rows"]
    symmetric = quantization["symmetric"]
    if type(symmetric) is not bool:
        raise QualificationError("record.quantization.symmetric must be a boolean")
    if scheme == "float8":
        if (
            type(group_size) is not int
            or group_size != 0
            or symmetric is not True
            or calibration is not None
            or overlap_rows is not None
        ):
            raise QualificationError(
                "float8 must declare group_size 0, symmetric true, null calibration, "
                "and null calibration overlap"
            )
    else:
        _integer(group_size, "record.quantization.group_size", 1)
        _dataset(calibration, "record.quantization.calibration")
        _integer(
            overlap_rows,
            "record.quantization.calibration_evaluation_overlap_rows",
        )
    if method not in METHODS or METHODS[method] != (scheme, symmetric):
        raise QualificationError(
            "record.quantization.method is inconsistent with scheme and symmetry"
        )

    evaluation = _mapping(record["evaluation"], "record.evaluation")
    _exact_keys(
        evaluation,
        "record.evaluation",
        {"dataset", "harness", "quality", "reliability", "memory", "performance"},
    )
    evaluated = _dataset(evaluation["dataset"], "record.evaluation.dataset")
    if calibration is not None:
        same_source = calibration["repo"] == evaluated["repo"]
        if calibration["digest"] == evaluated["digest"] or (
            same_source and _base_splits(calibration) & _base_splits(evaluated)
        ):
            raise QualificationError(
                "calibration and evaluation cannot reuse the same rows"
            )

    harness = _mapping(evaluation["harness"], "record.evaluation.harness")
    _exact_keys(
        harness,
        "record.evaluation.harness",
        {"code_revision", "command", "device", "dependencies"},
    )
    _revision(harness["code_revision"], "record.evaluation.harness.code_revision")
    command = harness["command"]
    if not isinstance(command, list) or not command:
        raise QualificationError(
            "record.evaluation.harness.command must be a non-empty array"
        )
    for index, part in enumerate(command):
        _text(part, f"record.evaluation.harness.command[{index}]")
    device = _mapping(harness["device"], "record.evaluation.harness.device")
    _exact_keys(
        device, "record.evaluation.harness.device", {"kind", "name", "architecture"}
    )
    if device["kind"] not in DEVICES:
        raise QualificationError(f"device kind must be one of {sorted(DEVICES)}")
    _text(device["name"], "record.evaluation.harness.device.name")
    _text(device["architecture"], "record.evaluation.harness.device.architecture")
    dependencies = _mapping(
        harness["dependencies"], "record.evaluation.harness.dependencies"
    )
    if not dependencies:
        raise QualificationError(
            "record.evaluation.harness.dependencies cannot be empty"
        )
    for name, version in dependencies.items():
        _text(name, "dependency name")
        _text(version, f"dependency {name!r}")

    rows = evaluated["rows"]
    _validate_quality(evaluation["quality"], rows)
    _validate_reliability(evaluation["reliability"])
    _validate_memory(evaluation["memory"])
    _validate_performance(evaluation["performance"], rows)

    gates = _mapping(record["gates"], "record.gates")
    _exact_keys(
        gates,
        "record.gates",
        {
            "min_decision_agreement",
            "max_accuracy_drop",
            "max_memory_ratio",
            "max_latency_ratio",
        },
    )
    _ratio(gates["min_decision_agreement"], "record.gates.min_decision_agreement")
    _ratio(gates["max_accuracy_drop"], "record.gates.max_accuracy_drop")
    _ratio(gates["max_memory_ratio"], "record.gates.max_memory_ratio")
    _number(gates["max_latency_ratio"], "record.gates.max_latency_ratio", 0.000001)


def _expected_outcome(record: dict[str, Any]) -> dict[str, Any]:
    evaluation = record["evaluation"]
    rows = evaluation["dataset"]["rows"]
    quality = evaluation["quality"]
    memory = evaluation["memory"]
    performance = evaluation["performance"]
    gates = record["gates"]
    reliability = evaluation["reliability"]
    reliability_failures = sum(reliability.values())
    overlap_rows = record["quantization"]["calibration_evaluation_overlap_rows"] or 0
    observed = {
        "decision_agreement": round(1 - quality["changed_answers"] / rows, 8),
        "accuracy_drop": round(
            max(0, quality["baseline_correct"] - quality["candidate_correct"]) / rows,
            8,
        ),
        "memory_ratio": round(
            memory["candidate_peak_bytes"] / memory["baseline_peak_bytes"], 8
        ),
        "latency_ratio": round(
            performance["latency_ms"]["candidate"]["median"]
            / performance["latency_ms"]["baseline"]["median"],
            8,
        ),
        "reliability_failures": reliability_failures,
        "calibration_overlap_rows": overlap_rows,
    }
    limits = {
        "decision_agreement": ("minimum", gates["min_decision_agreement"]),
        "accuracy_drop": ("maximum", gates["max_accuracy_drop"]),
        "memory_ratio": ("maximum", gates["max_memory_ratio"]),
        "latency_ratio": ("maximum", gates["max_latency_ratio"]),
        "reliability_failures": ("maximum", 0),
        "calibration_overlap_rows": ("maximum", 0),
    }
    checks = {}
    for name, value in observed.items():
        direction, limit = limits[name]
        passed = value >= limit if direction == "minimum" else value <= limit
        checks[name] = {"observed": value, direction: limit, "passed": passed}
    return {
        "passed": all(check["passed"] for check in checks.values()),
        "checks": checks,
    }


def finalize_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return the record with its pass/fail outcome computed from declared gates."""
    result = json.loads(json.dumps(record))
    result.pop("outcome", None)
    _validate_core(result)
    result["outcome"] = _expected_outcome(result)
    return result


def check_record(record: dict[str, Any]) -> None:
    """Reject a malformed record or a pass/fail outcome inconsistent with its evidence."""
    _validate_core(record)
    if "outcome" not in record:
        raise QualificationError("record.outcome is missing; finalize the record first")
    expected = _expected_outcome(record)
    if record["outcome"] != expected:
        raise QualificationError(
            "record.outcome differs from the evidence and declared gates"
        )


def verify_package_manifest(ref: dict[str, Any], path: Path, label: str) -> None:
    """Tie a record identity to the package manifest bytes without parsing weights."""
    try:
        raw = Path(path).read_bytes()
        manifest = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise QualificationError(f"{label} manifest cannot be read: {exc}") from exc
    if hashlib.sha256(raw).hexdigest() != ref["manifest_sha256"]:
        raise QualificationError(f"{label} manifest SHA-256 differs from the record")
    if not isinstance(manifest, dict) or manifest.get("schema") != MANIFEST_SCHEMA:
        raise QualificationError(f"{label} is not a {MANIFEST_SCHEMA} manifest")
    identity = manifest.get("identity")
    if (
        not isinstance(identity, dict)
        or identity.get("model_sha256") != ref["model_sha256"]
    ):
        raise QualificationError(f"{label} model identity differs from the record")


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise QualificationError(f"{path} cannot be read: {exc}") from exc
    return _mapping(value, str(path))


def _verify_manifests(args: argparse.Namespace, record: dict[str, Any]) -> None:
    for label in ("baseline", "candidate"):
        path = getattr(args, f"{label}_manifest")
        if path is not None:
            verify_package_manifest(record[label], path, label)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    finalize = commands.add_parser(
        "finalize", help="compute and write the qualification verdict"
    )
    finalize.add_argument("input", type=Path)
    finalize.add_argument("output", type=Path)
    check = commands.add_parser("check", help="validate a qualification record")
    check.add_argument("record", type=Path)
    for command in (finalize, check):
        command.add_argument("--baseline-manifest", type=Path)
        command.add_argument("--candidate-manifest", type=Path)
    args = parser.parse_args()

    if args.command == "finalize":
        if args.baseline_manifest is None or args.candidate_manifest is None:
            parser.error(
                "finalize requires --baseline-manifest and --candidate-manifest"
            )
        record = finalize_record(_read(args.input))
        _verify_manifests(args, record)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"{args.output}: {'PASS' if record['outcome']['passed'] else 'FAIL'}")
        return 0 if record["outcome"]["passed"] else 1

    record = _read(args.record)
    check_record(record)
    _verify_manifests(args, record)
    print(f"{args.record}: {'PASS' if record['outcome']['passed'] else 'FAIL'}")
    return 0 if record["outcome"]["passed"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except QualificationError as exc:
        print(f"qualification error: {exc}", file=sys.stderr)
        raise SystemExit(INVALID_RECORD) from exc
