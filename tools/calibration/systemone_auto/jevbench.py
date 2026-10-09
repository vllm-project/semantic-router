"""Serial public-suite collection: fixed gates, all failures, no fitting."""

from __future__ import annotations

import copy
import math
import os
import re
import statistics
import sys
import urllib.parse
import urllib.request
from http import HTTPStatus
from pathlib import Path

from .artifacts import canonical, digest, file_digest, read_json, write_json
from .collection import MAX_TIMEOUT_SECONDS, _NoRedirect, request_once
from .jevbench_suite import (
    ARMS,
    COMMIT,
    REVISIONS,
    THRESHOLD,
    load_suite,
    request_for,
    schedule_for,
    score_response,
    validate_schedule,
)

MAX_PASSES = 100
COUNTER_TOLERANCE = 1e-9


def add_parser(commands) -> None:
    parser = commands.add_parser(
        "jevbench",
        help="run pinned public JevBench through native direct/auto endpoints",
    )
    parser.add_argument("--jevbench-checkout", type=Path, required=True)
    parser.add_argument("--suite-commit", choices=[COMMIT], default=COMMIT)
    parser.add_argument(
        "--endpoint",
        required=True,
        help="base URL; /v1/systemone is appended; never stored",
    )
    parser.add_argument(
        "--key-env",
        default="",
        help="API-key environment variable; empty for no authentication",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="predeployed Router YAML; hash recorded, content never copied",
    )
    parser.add_argument(
        "--gate-threshold",
        type=float,
        default=THRESHOLD,
        help="predeclared deployed threshold; collector never selects or changes it",
    )
    parser.add_argument("--kai-model", default="kai")
    parser.add_argument("--vega-model", default="vega")
    parser.add_argument("--auto-model", default="vllm-sr/auto")
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument(
        "--schedule", type=Path, help="optional frozen public task/arm schedule JSON"
    )
    parser.add_argument(
        "--warmups", type=Path, help="optional frozen Choice/Noul/Score request objects"
    )
    parser.add_argument(
        "--kai-metrics", help="optional native /metrics URL; use both model flags"
    )
    parser.add_argument(
        "--vega-metrics", help="optional native /metrics URL; use both model flags"
    )
    parser.add_argument("--metrics-key-env", default="")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="write label-free plan, make no network requests",
    )


def _address(value: str) -> str:
    parsed = urllib.parse.urlparse(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "endpoint must be HTTP(S), without userinfo, query or fragment"
        )
    return value.rstrip("/")


def _secret(name: str) -> str | None:
    if not name:
        return None
    value = os.environ.get(name)
    if not value:
        raise ValueError("requested API-key environment variable is empty")
    return value


def _counters(url: str, key: str | None, timeout: float) -> dict:
    headers = {"Authorization": "Bearer " + key} if key else {}
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.build_opener(_NoRedirect).open(
        request, timeout=timeout
    ) as response:
        text = response.read(4 * 1024 * 1024 + 1)
    if len(text) > 4 * 1024 * 1024:
        raise ValueError("metric response too large")
    result = {}
    for field, name, endpoint_only in (
        ("physical_calls", "vllm_srun_requests_total", True),
        ("forward_seconds", "vllm_srun_forward_duration_seconds_sum", False),
    ):
        values = []
        for line in text.decode().splitlines():
            match = re.fullmatch(
                re.escape(name) + r"(\{[^\n]*\})?\s+(\S+)(?:\s+\S+)?", line
            )
            if match and (
                not endpoint_only or 'endpoint="/v1/systemone"' in (match[1] or "")
            ):
                number = float(match[2])
                if not math.isfinite(number) or number < 0:
                    raise ValueError("invalid native counter")
                values.append(number)
        if not values:
            raise ValueError("required native counter absent")
        result[field] = sum(values)
    return result


def metric_snapshot(endpoints: dict, key: str | None, timeout: float):
    if not endpoints:
        return None
    try:
        return {name: _counters(url, key, timeout) for name, url in endpoints.items()}
    except (OSError, ValueError, TimeoutError, UnicodeError):
        return None  # Unknown metrics never become zero model calls.


def metric_delta(before, after):
    if before is None or after is None:
        return None
    result = {
        name: {key: after[name][key] - value for key, value in fields.items()}
        for name, fields in before.items()
    }
    if any(
        value < -COUNTER_TOLERANCE
        for fields in result.values()
        for value in fields.values()
    ):
        return None  # Counter reset invalidates the delta, not the HTTP sample.
    return result


def safe_response(raw: dict, question: dict) -> dict:
    """Retain typed answer values; remove deployment details and error strings."""
    result = {}
    answers = raw.get("answers")
    if isinstance(answers, dict) and isinstance(answers.get("decision"), dict):
        answer = answers["decision"]
        clean = {}
        kind = question["type"]
        if isinstance(answer.get("type"), str) and answer["type"] in {
            "choice",
            "noul",
            "score",
        }:
            clean["type"] = answer["type"]
        labels = (
            set(question["criteria"])
            if kind == "choice"
            else {str(i) for i in range(len(question.get("criteria", [])))}
        )
        if isinstance(answer.get("choice"), str) and answer["choice"] in labels:
            clean["choice"] = answer["choice"]
        for key in ("noul", "score", "confidence"):
            value = answer.get(key)
            if type(value) in {int, float} and math.isfinite(value):
                clean[key] = value
        probabilities = answer.get("probabilities")
        if (
            isinstance(probabilities, dict)
            and set(probabilities) <= labels
            and all(
                type(v) in {int, float} and math.isfinite(v)
                for v in probabilities.values()
            )
        ):
            clean["probabilities"] = probabilities
        if isinstance(answer.get("input_coverage"), str) and answer[
            "input_coverage"
        ] in {"complete", "partial", "truncated"}:
            clean["input_coverage"] = answer["input_coverage"]
        if answer.get("error"):
            clean["error"] = {"code": "native_question_error"}
        result["answers"] = {"decision": clean}
    meta = raw.get("meta")
    if isinstance(meta, dict):
        cleaned = {}
        for key, length in (("revision", 40), ("model_sha256", 64)):
            if re.fullmatch(r"[0-9a-f]{" + str(length) + "}", str(meta.get(key, ""))):
                cleaned[key] = meta[key]
        for key, allowed in {
            "engine": {"native"},
            "profile": {"exact"},
            "numerics": {"exact"},
            "accelerator": {"cpu", "cuda", "rocm", "mps"},
        }.items():
            if isinstance(meta.get(key), str) and meta[key] in allowed:
                cleaned[key] = meta[key]
        for key in ("queue_ms", "compute_ms"):
            value = meta.get(key)
            if type(value) in {int, float} and math.isfinite(value) and value >= 0:
                cleaned[key] = value
        result["meta"] = cleaned
    if raw.get("error"):
        result["error"] = {"code": "native_error"}
    return result


def native_identity(raw: dict, identities: dict) -> str | None:
    meta = raw.get("meta", {})
    model = next(
        (
            name
            for name, revision in REVISIONS.items()
            if meta.get("revision") == revision
        ),
        None,
    )
    if model is None or not re.fullmatch(
        r"[0-9a-f]{64}", str(meta.get("model_sha256", ""))
    ):
        return None
    fields = (
        "revision",
        "model_sha256",
        "engine",
        "profile",
        "numerics",
        "accelerator",
    )
    identity = {key: meta.get(key) for key in fields}
    if any(not isinstance(value, str) or not value for value in identity.values()):
        return None
    if identity != identities.setdefault(model, identity):
        return None
    return model


def _quantile(values: list[float], probability: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * probability
    return values[math.floor(position)] * (1 - position % 1) + values[
        math.ceil(position)
    ] * (position % 1)


def summarize(rows: list[dict], planned_per_arm: int) -> dict:
    quality, passes = {}, {}
    for arm in ARMS:
        values = [row for row in rows if row["pass"] == 0 and row["arm"] == arm]
        quality[arm] = {
            "planned": planned_per_arm,
            "observed": len(values),
            "correct": sum(row["score"]["correct"] for row in values),
            "unresolved": planned_per_arm
            - sum(row["score"]["valid"] for row in values),
            "accuracy": sum(row["score"]["correct"] for row in values)
            / planned_per_arm,
            "kai_coverage": sum(
                row["score"]["valid"] and row["selected_model"] == "kai"
                for row in values
            )
            / planned_per_arm,
        }
    for index in sorted({row["pass"] for row in rows}):
        arms = {}
        for arm in ARMS:
            values = [row for row in rows if row["pass"] == index and row["arm"] == arm]
            if not values:
                continue
            elapsed = [row["client_elapsed_ms"] for row in values]
            known = all(row["native_metrics_delta"] is not None for row in values)
            arms[arm] = {
                "requests": len(values),
                "http_failures": sum(
                    row["http_status"] != HTTPStatus.OK for row in values
                ),
                "elapsed_ms": {
                    "mean": statistics.mean(elapsed),
                    "p50": _quantile(elapsed, 0.5),
                    "p95": _quantile(elapsed, 0.95),
                    "p99": _quantile(elapsed, 0.99),
                },
                "physical_calls": (
                    {
                        name: sum(
                            row["native_metrics_delta"][name]["physical_calls"]
                            for row in values
                        )
                        for name in REVISIONS
                    }
                    if known
                    else None
                ),
                "metric_unknown_requests": sum(
                    row["native_metrics_delta"] is None for row in values
                ),
            }
        passes[str(index)] = arms
    return {
        "quality_pass": 0,
        "quality": quality,
        "passes": passes,
        "limits": [
            "Public 231-item suite, not official sealed v1.6.1 composite",
            "Serial frontend timing; repeated passes do not increase quality sample size",
            "All planned failures remain in the quality denominator; timing includes errors",
            "Counter deltas require exclusive runtimes; unknown counters stay null",
            "No equal-resource throughput, GPU-active time or provisioned cost claim",
        ],
    }


def run(args) -> dict:
    if (
        not math.isfinite(args.gate_threshold)
        or not 0 <= args.gate_threshold <= 1
        or not 0 < args.timeout <= MAX_TIMEOUT_SECONDS
        or not 1 <= args.passes <= MAX_PASSES
    ):
        raise ValueError("threshold, timeout or passes outside supported bounds")
    endpoint = _address(args.endpoint) + "/v1/systemone"
    models = dict(
        zip(ARMS, (args.kai_model, args.vega_model, args.auto_model), strict=True)
    )
    if len(set(models.values())) != len(ARMS) or not all(models.values()):
        raise ValueError("three distinct nonempty public model aliases are required")
    if bool(args.kai_metrics) != bool(args.vega_metrics):
        raise ValueError("supply both native metrics endpoints or neither")
    endpoints = (
        {
            name: _address(value)
            for name, value in (("kai", args.kai_metrics), ("vega", args.vega_metrics))
        }
        if args.kai_metrics
        else {}
    )
    key, metric_key = (
        (None, None)
        if args.plan_only
        else (_secret(args.key_env), _secret(args.metrics_key_env))
    )
    suite = load_suite(args.jevbench_checkout, args.suite_commit)
    schedule = (
        read_json(args.schedule)
        if args.schedule
        else schedule_for(suite.tasks, args.passes, args.seed)
    )
    validate_schedule(schedule, suite.tasks, args.passes)
    if args.warmups:
        warmups = read_json(args.warmups)
    else:
        warmups = {}
        for task in sorted(suite.tasks.values(), key=lambda task: task.id):
            kind = task.question["type"]
            warmups.setdefault(kind, request_for(suite, task, args.kai_model))
    if set(warmups) != {"choice", "noul", "score"}:
        raise ValueError("exactly three typed warmup requests are required")
    for kind, payload in warmups.items():
        if (
            set(payload["questions"]) != {"decision"}
            or payload["questions"]["decision"]["type"] != kind
            or payload["questions"]["decision"].get("require_full_input") is not True
            or payload.get("options", {}).get("return_meta") is not True
        ):
            raise ValueError(
                "warmups must use the same full-input/meta request contract"
            )
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise ValueError(
            "output directory must be empty; never silently retry a partial run"
        )
    output.mkdir(parents=True, exist_ok=True)
    plan = {
        "schema_version": "systemone-jevbench-run/v1",
        "suite": suite.source,
        "model_aliases": models,
        "native_revisions": REVISIONS,
        "router_config_sha256": file_digest(args.config),
        "declared_gate_threshold": args.gate_threshold,
        "gate_notice": "Operator-declared deployed gate; collector neither edits nor selects it. Config hash is not proof of remote readback.",
        "source_groups": len({task.group or task.id for task in suite.tasks.values()}),
        "passes": args.passes,
        "quality_pass": 0,
        "concurrency": 1,
        "retries": 0,
        "timeout_seconds": args.timeout,
        "request_order": schedule,
        "warmups": warmups,
        "metrics_requested": bool(endpoints),
        "warmup_order": ["choice", "noul", "score"],
        "tool_source_sha256": {
            name: file_digest(Path(sys.modules["systemone_auto." + name].__file__))
            for name in (
                "jevbench",
                "jevbench_suite",
                "collection",
                "metrics",
                "artifacts",
            )
        },
        "request_options": {
            "require_full_input_per_question": True,
            "return_meta": True,
        },
        "response_projection": "Native answer values and allowlisted model metadata; error messages, device and deployment details omitted.",
        "request_sha256": {
            task_id: digest(request_for(suite, task, args.auto_model))
            for task_id, task in suite.tasks.items()
        },
    }
    identity = digest(plan)
    write_json(output / "plan.json", plan)
    rows, identities = [], {}
    receipt = {
        "complete": False,
        "plan_only": args.plan_only,
        "run_identity": identity,
        "expected_count": len(schedule),
        "observation_count": 0,
        "warmup_count": 0,
    }
    write_json(output / "collection.json", receipt)
    if args.plan_only:
        return receipt

    def invoke(payload, item, phase):
        before = metric_snapshot(endpoints, metric_key, args.timeout)
        status, response, elapsed = request_once(endpoint, key, payload, args.timeout)
        after = metric_snapshot(endpoints, metric_key, args.timeout)
        raw = safe_response(response, payload["questions"]["decision"])
        public_alias_valid = response.get("model") == payload["model"]
        selected = (
            native_identity(raw, identities)
            if status == HTTPStatus.OK and public_alias_valid
            else None
        )
        row = {
            **item,
            "phase": phase,
            "run_identity": identity,
            "request_sha256": digest(payload),
            "http_status": status,
            "client_elapsed_ms": elapsed,
            "native_metrics_delta": metric_delta(before, after),
            "selected_model": selected,
            "public_alias_valid": public_alias_valid,
            "raw_response": raw,
        }
        if phase == "measured":
            task = suite.tasks[item["task_id"]]
            score = score_response(suite, task, payload, status, raw)
            expected_model = {"direct_kai": "kai", "direct_vega": "vega"}.get(
                item["arm"]
            )
            if selected is None or (expected_model and selected != expected_model):
                score = {"valid": False, "correct": False, "top_probability": None}
            row.update(source_group=task.group or task.id, score=score)
        path = output / ("warmup.jsonl" if phase == "warmup" else "observations.jsonl")
        with path.open("a") as stream:
            stream.write(canonical(row) + "\n")
            stream.flush()
        return row

    try:
        for position, kind in enumerate(plan["warmup_order"]):
            original = warmups[kind]
            for arm_position, arm in enumerate(ARMS):
                payload = copy.deepcopy(original)
                payload["model"] = models[arm]
                invoke(
                    payload,
                    {
                        "task_id": "warmup-" + kind,
                        "pass": -1,
                        "position": position,
                        "arm": arm,
                        "arm_position": arm_position,
                    },
                    "warmup",
                )
                receipt["warmup_count"] += 1
        for item in schedule:
            payload = request_for(
                suite, suite.tasks[item["task_id"]], models[item["arm"]]
            )
            rows.append(invoke(payload, item, "measured"))
    finally:
        receipt.update(
            observation_count=len(rows),
            complete=len(rows) == len(schedule),
            runtime_identities=identities,
            files={
                path.name: file_digest(path)
                for path in output.iterdir()
                if path.suffix == ".jsonl"
            },
        )
        write_json(output / "collection.json", receipt)
        write_json(output / "summary.json", summarize(rows, len(suite.tasks)))
    return receipt
