"""Run synthetic Decision workflows and collate six-model semantic receipts."""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import math
import os
import random
import re
import subprocess
import time
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from .cases import MODELS
from .semantic_audit import audit_cohorts
from .semantic_cases import WorkloadCase, cohort_sha256, generate_cases
from .semantic_metrics import (
    MetricCapture,
    MetricsError,
    read_snapshot,
    summarize_captures,
    telemetry_comparison,
    validate_metrics_url,
)
from .semantic_report import (
    SCHEMA_VERSION,
    WorkflowSample,
    build_semantic_matrix,
    group_workflows,
    summarize_shape,
)
from .semantic_transport import HttpSample, batch_url, measure_http
from .transport import Endpoint, validate_endpoint_url

ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILES = (
    "__init__.py",
    "__main__.py",
    "cases.py",
    "legacy_projection.py",
    "transport.py",
    "report.py",
    "semantic_cases.py",
    "semantic_audit.py",
    "semantic_metrics.py",
    "semantic_transport.py",
    "semantic_report.py",
    "semantic_runner.py",
)
CONTRACT_FILES = (
    ROOT / "src/vllm-sr/decision_runtime/contracts.py",
    ROOT / "src/vllm-sr/decision_runtime/confidence.py",
)
SOURCE_LABEL = re.compile(r"(?:[0-9a-f]{40}|sha256:[0-9a-f]{64})\Z")
MODEL_REVISION = re.compile(r"[0-9a-f]{40}\Z")
HARDWARE_LABEL = re.compile(r"[A-Za-z0-9 ._+()-]{1,128}\Z")
MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}\Z")
MAX_TIMED_SEMANTIC_EVIDENCE_BYTES = 80 * 1024 * 1024
MAX_TIMED_SEMANTIC_COMPRESSED_BYTES = 16 * 1024 * 1024
MAX_TIMED_SEMANTIC_REQUEST_BYTES = 1024 * 1024


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _nonnegative_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a nonnegative integer") from error
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def _positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be positive and finite") from error
    if number <= 0 or not math.isfinite(number):
        raise argparse.ArgumentTypeError("must be positive and finite")
    return number


def _nonnegative_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be nonnegative and finite") from error
    if number < 0 or not math.isfinite(number):
        raise argparse.ArgumentTypeError("must be nonnegative and finite")
    return number


def _positive_csv(value: str) -> tuple[int, ...]:
    try:
        numbers = tuple(int(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected comma-separated positive integers"
        ) from error
    if (
        not numbers
        or any(number < 1 for number in numbers)
        or len(set(numbers)) != len(numbers)
    ):
        raise argparse.ArgumentTypeError("expected unique positive integers")
    return numbers


def _metadata(args: argparse.Namespace, arm: str) -> dict[str, str | int]:
    source_ref = getattr(args, f"{arm}_source_ref")
    model_revision = getattr(args, f"{arm}_model_revision")
    hardware = getattr(args, f"{arm}_hardware")
    if not SOURCE_LABEL.fullmatch(source_ref):
        raise ValueError(f"{arm} source ref must be a commit SHA or content digest")
    if not MODEL_REVISION.fullmatch(model_revision):
        raise ValueError(f"{arm} model revision must be a 40-character commit SHA")
    if not HARDWARE_LABEL.fullmatch(hardware) or hardware.lower() == "unknown":
        raise ValueError(f"{arm} hardware must be a public-safe model/count label")
    return {
        "source_ref": source_ref,
        "model_revision": model_revision,
        "hardware": hardware,
        "network_scope": getattr(args, f"{arm}_network_scope"),
        "declared_physical_batch_size": getattr(args, f"{arm}_physical_batch_size"),
    }


def _token(env_name: str | None) -> str | None:
    if env_name is None:
        return None
    token = os.environ.get(env_name)
    if not token:
        raise ValueError("requested token environment variable is unset or empty")
    return token


def _harness_digest() -> str:
    digest = hashlib.sha256()
    paths = [Path(__file__).with_name(name) for name in SOURCE_FILES]
    for path in (*paths, *CONTRACT_FILES):
        digest.update(str(path.relative_to(ROOT)).encode("utf-8"))
        digest.update(b"\x00")
        digest.update(path.read_bytes())
        digest.update(b"\x00")
    return digest.hexdigest()


def _source_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _schedule(
    cases: tuple[WorkloadCase, ...], count: int, seed: int
) -> list[WorkloadCase]:
    ordered = list(cases)
    random.Random(seed).shuffle(ordered)
    return [ordered[index % len(ordered)] for index in range(count)]


def _run_wave(
    arm: str,
    phase: str,
    round_number: int,
    selected: list[WorkloadCase],
    *,
    old: Endpoint,
    new_single: Endpoint,
    new_batch: Endpoint,
    concurrency: int,
    timeout: float,
    capture_timed_semantics: bool = False,
) -> tuple[list[HttpSample], list[WorkflowSample]]:
    jobs = []
    for sequence, case in enumerate(selected):
        if arm == "old":
            jobs.extend((sequence, case.id, old, spec) for spec in case.old_singles)
        else:
            endpoint = new_single if case.state_count == 1 else new_batch
            jobs.append((sequence, case.id, endpoint, case.new_request))
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                measure_http,
                endpoint,
                spec,
                case_id=case_id,
                concurrency=concurrency,
                phase=phase,
                round_number=round_number,
                sequence=sequence,
                timeout_seconds=timeout,
                capture_response=(
                    capture_timed_semantics
                    and arm == "new"
                    and phase == "throughput"
                    and concurrency in (8, 32)
                ),
            )
            for sequence, case_id, endpoint, spec in jobs
        ]
        samples = [future.result() for future in as_completed(futures)]
    samples.sort(key=lambda item: (item.sequence, item.state_id or ""))
    return samples, group_workflows(
        arm, phase, round_number, selected, samples, concurrency
    )


def _logical_schedule_sha256(
    selected: list[WorkloadCase], phase: str, round_number: int, concurrency: int
) -> str:
    """Identify equal ordered logical cases, not a paired timed-arrival epoch."""

    payload = {
        "question_count": selected[0].question_count,
        "state_count": selected[0].state_count,
        "concurrency": concurrency,
        "phase": phase,
        "round": round_number,
        "case_ids": [case.id for case in selected],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _record_wave(
    samples_handle,
    workflows_handle,
    samples,
    workflows,
    selected,
    origin_ns,
    timed_handle=None,
    timed_bytes=None,
):
    schedule_sha256 = _logical_schedule_sha256(
        selected,
        workflows[0].phase,
        workflows[0].round,
        workflows[0].concurrency,
    )
    for sample in samples:
        record = sample.public_record(origin_ns)
        record["logical_schedule_sha256"] = schedule_sha256
        samples_handle.write(json.dumps(record, sort_keys=True) + "\n")
        if sample.response_body is not None:
            if timed_handle is None or timed_bytes is None:
                raise ValueError("timed semantic response has no evidence writer")
            if (
                sample.request_body is None
                or len(sample.request_body) > MAX_TIMED_SEMANTIC_REQUEST_BYTES
            ):
                raise ValueError("timed semantic request exceeds the bounded archive")
            evidence = {
                "case_id": sample.case_id,
                "concurrency": sample.concurrency,
                "round": sample.round,
                "sequence": sample.sequence,
                "request_sha256": sample.request_sha256,
                "request_base64": base64.b64encode(sample.request_body).decode("ascii"),
                "response_sha256": sample.response_sha256,
                "response_base64": base64.b64encode(sample.response_body).decode(
                    "ascii"
                ),
            }
            line = json.dumps(evidence, sort_keys=True, separators=(",", ":")) + "\n"
            timed_bytes[0] += len(line.encode("utf-8"))
            if timed_bytes[0] > MAX_TIMED_SEMANTIC_EVIDENCE_BYTES:
                raise ValueError("timed semantic evidence exceeds the bounded archive")
            timed_handle.write(line)
    for workflow in workflows:
        workflows_handle.write(
            json.dumps(workflow.public_record(origin_ns), sort_keys=True) + "\n"
        )
    samples_handle.flush()
    workflows_handle.flush()


def run_semantic(args: argparse.Namespace) -> int:
    old_url = validate_endpoint_url(args.old_url)
    new_url = validate_endpoint_url(args.new_url)
    old_meta = _metadata(args, "old")
    new_meta = _metadata(args, "new")
    old_model_id = args.old_model_id or args.model
    if not MODEL_ID.fullmatch(old_model_id):
        raise ValueError("old model ID is not a public-safe model slug")
    old = Endpoint("old", old_url, _token(args.old_token_env), args.old_response_mode)
    new_single = Endpoint("new", new_url, _token(args.new_token_env))
    new_batch = Endpoint("new", batch_url(new_url), new_single.token)
    metrics_urls = {
        "old": (
            validate_metrics_url(args.old_metrics_url) if args.old_metrics_url else None
        ),
        "new": (
            validate_metrics_url(args.new_metrics_url) if args.new_metrics_url else None
        ),
    }
    cohorts = {
        (questions, states): generate_cases(
            args.model,
            old_model_id,
            question_count=questions,
            state_count=states,
            variants=args.variants,
            seed=args.seed,
        )
        for questions in args.question_counts
        for states in args.state_counts
    }
    settings = {
        "question_counts": args.question_counts,
        "state_counts": args.state_counts,
        "concurrencies": args.concurrencies,
        "variants_per_shape": args.variants,
        "seed": args.seed,
        "warmup_workflows_per_arm": args.warmup,
        "latency_workflows_per_arm": args.latency_workflows,
        "throughput_workflows_per_round_per_arm": args.throughput_workflows,
        "throughput_rounds": args.rounds,
        "timeout_seconds": args.timeout,
        "parity_policy": args.parity_policy,
        "timed_semantic_evidence": args.timed_semantic_evidence,
        "probability_tolerance_absolute": args.probability_tolerance,
        "concurrency_unit": "maximum in-flight HTTP requests per arm",
        "arrival_policy": (
            "the same ordered logical workflows are submitted per arm and round; "
            "old fan-out submits one HTTP call per state while new batch submits "
            "one HTTP call per workflow; submission and server admission times "
            "are not paired across arms"
        ),
        "connection_policy": "new HTTP connection per request",
        "latency_boundary": "first HTTP send through last complete response body in workflow",
        "percentile": "nearest rank over complete conforming workflows",
        "metrics_collection": {
            "old": metrics_urls["old"] is not None,
            "new": metrics_urls["new"] is not None,
            "boundary": "before and after each throughput wave, outside HTTP timing",
        },
    }
    cohort_digest = hashlib.sha256()
    for cases in cohorts.values():
        cohort_digest.update(cohort_sha256(cases).encode("ascii"))
        cohort_digest.update(b"\x00")
    adapter_kind = "none"
    if old_model_id != args.model:
        adapter_kind = "model_id_only"
    if args.old_response_mode == "legacy_preview":
        adapter_kind = (
            "legacy_preview_and_model_id"
            if old_model_id != args.model
            else "legacy_preview"
        )
    receipt_base = {
        "schema_version": SCHEMA_VERSION,
        "scope": "synthetic Decision HTTP workflow performance and untimed semantic/token parity; no eval-quality score",
        "model": args.model,
        "old": old_meta,
        "new": new_meta,
        "adapter": {
            "kind": adapter_kind,
            "old_model_id": old_model_id,
            "envelope_transform": (
                "old_max_probability_to_decision_v1_for_validation"
                if args.old_response_mode == "legacy_preview"
                else "none"
            ),
            "applied_outside_timed_interval": True,
        },
        "source_commit": _source_commit(),
        "harness_sha256": _harness_digest(),
        "generator_sha256": hashlib.sha256(
            Path(__file__).with_name("semantic_cases.py").read_bytes()
        ).hexdigest(),
        "cohort_sha256": cohort_digest.hexdigest(),
        "case_ids": [case.id for cases in cohorts.values() for case in cases],
        "settings": settings,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "audit.jsonl").open("w", encoding="utf-8") as audit_handle:
        audit = audit_cohorts(
            cohorts,
            old,
            new_single,
            new_batch,
            timeout=args.timeout,
            probability_tolerance=args.probability_tolerance,
            output_handle=audit_handle,
        )
    audit["policy"] = args.parity_policy
    audit["provenance"] = {
        "old": old_meta,
        "new": new_meta,
        "old_model_id": old_model_id,
        "new_model_id": args.model,
        "cohort_sha256": receipt_base["cohort_sha256"],
    }
    audit["prompt_comparison_scope"] = (
        "identical logical state and questions except model ID and batch envelope; "
        "server-side prompt rendering is not observable"
    )
    audit["quality_accuracy_claimed"] = False
    audit["comparison_eligible"] = (
        audit["status"] == "passed" and args.parity_policy == "require"
    )
    if audit["status"] != "passed" and args.parity_policy == "require":
        receipt_path = args.output_dir / "receipt.json"
        receipt = {
            **receipt_base,
            "measured_at_utc": None,
            "status": "audit_failed",
            "audit": audit,
            "shapes": [],
            "failed_workflows": 0,
            "failed_metrics_shapes": 0,
        }
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        print(
            json.dumps(
                {
                    "receipt": str(receipt_path),
                    "audit_status": "failed",
                    "measured": False,
                }
            )
        )
        return 1

    origin_ns = time.perf_counter_ns()
    shape_rows = []
    total_errors = 0
    total_metric_errors = 0
    timed_context = (
        gzip.open(args.output_dir / "timed-semantic.jsonl.gz", "wt", encoding="utf-8")
        if args.timed_semantic_evidence
        else nullcontext(None)
    )
    timed_bytes = [0]
    with (
        (args.output_dir / "samples.jsonl").open(
            "w", encoding="utf-8"
        ) as sample_handle,
        (args.output_dir / "workflows.jsonl").open(
            "w", encoding="utf-8"
        ) as workflow_handle,
        (args.output_dir / "metrics.jsonl").open(
            "w", encoding="utf-8"
        ) as metrics_handle,
        timed_context as timed_handle,
    ):
        for (questions, states), cases in cohorts.items():
            for concurrency in args.concurrencies:
                samples: list[HttpSample] = []
                workflows: list[WorkflowSample] = []
                captures: list[MetricCapture] = []
                warmup = _schedule(cases, args.warmup, args.seed)
                latency = _schedule(cases, args.latency_workflows, args.seed + 1)
                for phase, selected in (("warmup", warmup), ("latency", latency)):
                    for sequence, case in enumerate(selected):
                        order = ("old", "new") if sequence % 2 == 0 else ("new", "old")
                        for arm in order:
                            wave_samples, wave_workflows = _run_wave(
                                arm,
                                phase,
                                sequence,
                                [case],
                                old=old,
                                new_single=new_single,
                                new_batch=new_batch,
                                concurrency=concurrency,
                                timeout=args.timeout,
                                capture_timed_semantics=args.timed_semantic_evidence,
                            )
                            samples.extend(wave_samples)
                            workflows.extend(wave_workflows)
                            _record_wave(
                                sample_handle,
                                workflow_handle,
                                wave_samples,
                                wave_workflows,
                                [case],
                                origin_ns,
                                timed_handle,
                                timed_bytes,
                            )
                for round_number in range(args.rounds):
                    selected = _schedule(
                        cases, args.throughput_workflows, args.seed + 2 + round_number
                    )
                    order = ("old", "new") if round_number % 2 == 0 else ("new", "old")
                    for arm in order:
                        metrics_url = metrics_urls[arm]
                        metrics_token = old.token if arm == "old" else new_single.token
                        metrics_model = old_model_id if arm == "old" else args.model
                        before = None
                        metrics_error = None
                        if metrics_url is not None:
                            try:
                                before = read_snapshot(
                                    metrics_url,
                                    metrics_token,
                                    metrics_model,
                                    args.timeout,
                                )
                            except MetricsError as error:
                                metrics_error = str(error)
                        wave_samples, wave_workflows = _run_wave(
                            arm,
                            "throughput",
                            round_number,
                            selected,
                            old=old,
                            new_single=new_single,
                            new_batch=new_batch,
                            concurrency=concurrency,
                            timeout=args.timeout,
                            capture_timed_semantics=args.timed_semantic_evidence,
                        )
                        if metrics_url is not None:
                            after = None
                            try:
                                after = read_snapshot(
                                    metrics_url,
                                    metrics_token,
                                    metrics_model,
                                    args.timeout,
                                )
                            except MetricsError as error:
                                metrics_error = metrics_error or str(error)
                            capture = MetricCapture(
                                arm=arm,
                                question_count=questions,
                                state_count=states,
                                concurrency=concurrency,
                                round=round_number,
                                before=before,
                                after=after,
                                error_code=metrics_error,
                            )
                            captures.append(capture)
                            metrics_handle.write(
                                json.dumps(capture.public_record(), sort_keys=True)
                                + "\n"
                            )
                            metrics_handle.flush()
                        samples.extend(wave_samples)
                        workflows.extend(wave_workflows)
                        _record_wave(
                            sample_handle,
                            workflow_handle,
                            wave_samples,
                            wave_workflows,
                            selected,
                            origin_ns,
                            timed_handle,
                            timed_bytes,
                        )
                summary = summarize_shape(
                    workflows,
                    samples,
                    old_meta,
                    new_meta,
                    state_count=states,
                    same_wire_bytes=all(case.same_wire_bytes for case in cases),
                )
                if not audit["comparison_eligible"]:
                    comparison = summary["comparison"]
                    comparison["eligible"] = False
                    comparison["reasons"].append("exploratory_parity_report_mode")
                    if audit["status"] != "passed":
                        comparison["reasons"].append("semantic_token_parity_failed")
                    comparison["old_over_new_p50_workflow_latency"] = None
                    comparison["new_over_old_successful_decisions_per_second"] = None
                arm_telemetry = {
                    arm: summarize_captures(
                        [capture for capture in captures if capture.arm == arm],
                        requested=metrics_urls[arm] is not None,
                        successful_decisions=summary["arms"][arm]["throughput"][
                            "successful_decisions"
                        ],
                        failed_workflows=summary["arms"][arm]["throughput"][
                            "failed_workflows"
                        ],
                    )
                    for arm in ("old", "new")
                }
                summary["telemetry"] = {
                    "arms": arm_telemetry,
                    "comparison": telemetry_comparison(
                        arm_telemetry["old"], arm_telemetry["new"]
                    ),
                }
                if not audit["comparison_eligible"]:
                    summary["telemetry"]["comparison"] = {
                        "available": False,
                        "reason": "semantic_token_parity_not_qualified",
                        "old_over_new_row_preparation_seconds_per_decision": None,
                        "new_over_old_observed_rows_per_physical_batch": None,
                    }
                total_metric_errors += sum(
                    metrics_urls[arm] is not None
                    and arm_telemetry[arm]["status"] != "complete"
                    for arm in ("old", "new")
                )
                total_errors += sum(not item.success for item in workflows)
                shape_rows.append(
                    {
                        "question_count": questions,
                        "state_count": states,
                        "concurrency": concurrency,
                        "case_ids": [case.id for case in cases],
                        "summary": summary,
                    }
                )
    if (
        args.timed_semantic_evidence
        and (args.output_dir / "timed-semantic.jsonl.gz").stat().st_size
        > MAX_TIMED_SEMANTIC_COMPRESSED_BYTES
    ):
        raise ValueError("timed semantic compressed archive exceeds 16 MiB")
    receipt = {
        **receipt_base,
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "measured",
        "audit": audit,
        "shapes": shape_rows,
        "failed_workflows": total_errors,
        "failed_metrics_shapes": total_metric_errors,
    }
    receipt_path = args.output_dir / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "receipt": str(receipt_path),
                "failed_workflows": total_errors,
                "failed_metrics_shapes": total_metric_errors,
                "audit_status": audit["status"],
                "comparison_eligible": audit["comparison_eligible"],
            }
        )
    )
    return int(total_errors > 0 or total_metric_errors > 0)


def run_semantic_matrix(args: argparse.Namespace) -> int:
    matrix = build_semantic_matrix(args.receipts)
    output = json.dumps(matrix, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(output, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(output)
        print(args.output)
    return 0


def add_parsers(commands: argparse._SubParsersAction) -> None:
    from .semantic_arrivals import add_arrival_parser

    run = commands.add_parser(
        "semantic", help="measure synthetic mixed-question and multi-state workflows"
    )
    run.add_argument("--model", required=True, choices=MODELS)
    run.add_argument("--old-url", required=True)
    run.add_argument("--new-url", required=True)
    run.add_argument("--old-token-env")
    run.add_argument("--new-token-env")
    run.add_argument("--old-metrics-url", help="optional old /metrics endpoint")
    run.add_argument("--new-metrics-url", help="optional new /metrics endpoint")
    run.add_argument(
        "--old-model-id", help="audited model-ID translation for old singles"
    )
    run.add_argument(
        "--old-response-mode",
        choices=("decision_v1", "legacy_preview"),
        default="decision_v1",
        help="Validate the old preview envelope with an audited post-timing projection.",
    )
    for arm in ("old", "new"):
        run.add_argument(f"--{arm}-source-ref", required=True)
        run.add_argument(f"--{arm}-model-revision", required=True)
        run.add_argument(f"--{arm}-hardware", required=True)
        run.add_argument(
            f"--{arm}-network-scope",
            required=True,
            choices=("loopback", "private", "public"),
        )
        run.add_argument(
            f"--{arm}-physical-batch-size", required=True, type=_positive_int
        )
    run.add_argument("--question-counts", type=_positive_csv, default=(1, 8, 32))
    run.add_argument("--state-counts", type=_positive_csv, default=(1, 8, 32))
    run.add_argument("--concurrencies", type=_positive_csv, default=(1, 8, 32))
    run.add_argument("--variants", type=_positive_int, default=4)
    run.add_argument("--warmup", type=_nonnegative_int, default=2)
    run.add_argument("--latency-workflows", type=_positive_int, default=16)
    run.add_argument("--throughput-workflows", type=_positive_int, default=32)
    run.add_argument("--rounds", type=_positive_int, default=2)
    run.add_argument("--seed", type=int, default=17)
    run.add_argument("--timeout", type=_positive_float, default=60.0)
    run.add_argument(
        "--parity-policy",
        choices=("require", "report"),
        default="require",
        help="Require parity before timing, or measure exploratory ineligible ratios.",
    )
    run.add_argument(
        "--probability-tolerance",
        type=_nonnegative_float,
        default=0.01,
        help="Absolute old/new probability tolerance; default 0.01.",
    )
    run.add_argument(
        "--timed-semantic-evidence",
        action="store_true",
        help="Retain bounded exact request/response bytes from every new c8/c32 throughput wave.",
    )
    run.add_argument("--output-dir", required=True, type=Path)
    run.set_defaults(handler=run_semantic)

    matrix = commands.add_parser(
        "semantic-matrix", help="collate six semantic-workload receipts"
    )
    matrix.add_argument("--receipts", nargs="+", required=True, type=Path)
    matrix.add_argument("--output", type=Path)
    matrix.set_defaults(handler=run_semantic_matrix)
    add_arrival_parser(commands)
