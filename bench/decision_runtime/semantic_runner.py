"""Run synthetic Decision workflows and collate six-model semantic receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from .cases import MODELS
from .semantic_cases import WorkloadCase, cohort_sha256, generate_cases
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
    "transport.py",
    "report.py",
    "semantic_cases.py",
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
            )
            for sequence, case_id, endpoint, spec in jobs
        ]
        samples = [future.result() for future in as_completed(futures)]
    samples.sort(key=lambda item: (item.sequence, item.state_id or ""))
    return samples, group_workflows(
        arm, phase, round_number, selected, samples, concurrency
    )


def _record_wave(samples_handle, workflows_handle, samples, workflows, origin_ns):
    for sample in samples:
        samples_handle.write(
            json.dumps(sample.public_record(origin_ns), sort_keys=True) + "\n"
        )
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
    old = Endpoint("old", old_url, _token(args.old_token_env))
    new_single = Endpoint("new", new_url, _token(args.new_token_env))
    new_batch = Endpoint("new", batch_url(new_url), new_single.token)
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
        "concurrency_unit": "maximum in-flight HTTP requests per arm",
        "connection_policy": "new HTTP connection per request",
        "latency_boundary": "first HTTP send through last complete response body in workflow",
        "percentile": "nearest rank over complete conforming workflows",
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    origin_ns = time.perf_counter_ns()
    shape_rows = []
    total_errors = 0
    with (
        (args.output_dir / "samples.jsonl").open(
            "w", encoding="utf-8"
        ) as sample_handle,
        (args.output_dir / "workflows.jsonl").open(
            "w", encoding="utf-8"
        ) as workflow_handle,
    ):
        for (questions, states), cases in cohorts.items():
            for concurrency in args.concurrencies:
                samples: list[HttpSample] = []
                workflows: list[WorkflowSample] = []
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
                            )
                            samples.extend(wave_samples)
                            workflows.extend(wave_workflows)
                            _record_wave(
                                sample_handle,
                                workflow_handle,
                                wave_samples,
                                wave_workflows,
                                origin_ns,
                            )
                for round_number in range(args.rounds):
                    selected = _schedule(
                        cases, args.throughput_workflows, args.seed + 2 + round_number
                    )
                    order = ("old", "new") if round_number % 2 == 0 else ("new", "old")
                    for arm in order:
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
                        )
                        samples.extend(wave_samples)
                        workflows.extend(wave_workflows)
                        _record_wave(
                            sample_handle,
                            workflow_handle,
                            wave_samples,
                            wave_workflows,
                            origin_ns,
                        )
                summary = summarize_shape(
                    workflows,
                    samples,
                    old_meta,
                    new_meta,
                    state_count=states,
                    same_wire_bytes=all(case.same_wire_bytes for case in cases),
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
    cohort_digest = hashlib.sha256()
    for cases in cohorts.values():
        cohort_digest.update(cohort_sha256(cases).encode("ascii"))
        cohort_digest.update(b"\x00")
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "synthetic Decision HTTP workflow performance and strict response conformance; no eval-quality score",
        "model": args.model,
        "old": old_meta,
        "new": new_meta,
        "adapter": {
            "kind": "model_id_only" if old_model_id != args.model else "none",
            "old_model_id": old_model_id,
            "envelope_transform": "none",
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
        "shapes": shape_rows,
        "failed_workflows": total_errors,
    }
    receipt_path = args.output_dir / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"receipt": str(receipt_path), "failed_workflows": total_errors}))
    return int(total_errors > 0)


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
    run = commands.add_parser(
        "semantic", help="measure synthetic mixed-question and multi-state workflows"
    )
    run.add_argument("--model", required=True, choices=MODELS)
    run.add_argument("--old-url", required=True)
    run.add_argument("--new-url", required=True)
    run.add_argument("--old-token-env")
    run.add_argument("--new-token-env")
    run.add_argument(
        "--old-model-id", help="audited model-ID translation for old singles"
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
    run.add_argument("--output-dir", required=True, type=Path)
    run.set_defaults(handler=run_semantic)

    matrix = commands.add_parser(
        "semantic-matrix", help="collate six semantic-workload receipts"
    )
    matrix.add_argument("--receipts", nargs="+", required=True, type=Path)
    matrix.add_argument("--output", type=Path)
    matrix.set_defaults(handler=run_semantic_matrix)
