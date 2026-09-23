"""Run paired SystemOne measurements or collate six independent receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from .cases import DEFAULT_CASES, MODELS, Case, cohort_sha256, load_cases
from .report import SCHEMA_VERSION, build_matrix, summarize
from .semantic_runner import add_parsers as add_semantic_parsers
from .transport import Endpoint, Sample, measure, validate_endpoint_url

ROOT = Path(__file__).resolve().parents[2]
SOURCE_LABEL = re.compile(r"(?:[0-9a-f]{40}|sha256:[0-9a-f]{64})\Z")
HARDWARE_LABEL = re.compile(r"[A-Za-z0-9 ._+()-]{1,128}\Z")
MODEL_REVISION = re.compile(r"[0-9a-f]{40}\Z")
SOURCE_FILES = (
    "__init__.py",
    "__main__.py",
    "cases.py",
    "transport.py",
    "report.py",
    "cases.jsonl",
)
CONTRACT_FILES = (
    ROOT / "src/vllm-sr/decision_runtime/contracts.py",
    ROOT / "src/vllm-sr/decision_runtime/confidence.py",
)


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def _positive_float(value: str) -> float:
    number = float(value)
    if number <= 0 or not number < float("inf"):
        raise argparse.ArgumentTypeError("must be positive and finite")
    return number


def _metadata(args: argparse.Namespace, arm: str) -> dict[str, str]:
    source_ref = getattr(args, f"{arm}_source_ref")
    model_revision = getattr(args, f"{arm}_model_revision")
    hardware = getattr(args, f"{arm}_hardware")
    if not SOURCE_LABEL.fullmatch(source_ref):
        raise ValueError(f"{arm} source ref must be a commit SHA or content digest")
    if not MODEL_REVISION.fullmatch(model_revision):
        raise ValueError(
            f"{arm} model revision must be a 40-character lowercase commit SHA"
        )
    if not HARDWARE_LABEL.fullmatch(hardware) or hardware.lower() == "unknown":
        raise ValueError(f"{arm} hardware must be a public-safe model/count label")
    return {
        "source_ref": source_ref,
        "model_revision": model_revision,
        "hardware": hardware,
        "network_scope": getattr(args, f"{arm}_network_scope"),
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


def _schedule(cases: tuple[Case, ...], count: int, seed: int) -> list[Case]:
    ordered = list(cases)
    random.Random(seed).shuffle(ordered)
    return [ordered[index % len(ordered)] for index in range(count)]


def _save_sample(handle, sample: Sample, origin_ns: int) -> None:
    handle.write(json.dumps(sample.public_record(origin_ns), sort_keys=True) + "\n")
    handle.flush()


def _measure_one(
    endpoint: Endpoint,
    case: Case,
    phase: str,
    round_number: int,
    sequence: int,
    timeout: float,
) -> Sample:
    return measure(
        endpoint,
        case,
        phase=phase,
        round_number=round_number,
        sequence=sequence,
        timeout_seconds=timeout,
    )


def _run(args: argparse.Namespace) -> int:
    old_url = validate_endpoint_url(args.old_url)
    new_url = validate_endpoint_url(args.new_url)
    old_meta = _metadata(args, "old")
    new_meta = _metadata(args, "new")
    old = Endpoint("old", old_url, _token(args.old_token_env))
    new = Endpoint("new", new_url, _token(args.new_token_env))
    cases = load_cases(args.cases, args.model)
    schedule = _schedule(
        cases, max(args.warmup, args.latency_pairs, args.throughput_requests), args.seed
    )
    settings = {
        "warmup_per_arm": args.warmup,
        "latency_pairs": args.latency_pairs,
        "throughput_requests_per_round_per_arm": args.throughput_requests,
        "throughput_rounds": args.rounds,
        "concurrency": args.concurrency,
        "seed": args.seed,
        "timeout_seconds": args.timeout,
        "connection_policy": "new HTTP connection per request",
        "latency_boundary": "immediately before HTTP send through complete response body read",
        "percentile": "nearest rank over successful requests",
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    samples: list[Sample] = []
    origin_ns = time.perf_counter_ns()
    with (args.output_dir / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for sequence in range(args.warmup):
            for endpoint in (old, new) if sequence % 2 == 0 else (new, old):
                sample = _measure_one(
                    endpoint, schedule[sequence], "warmup", 0, sequence, args.timeout
                )
                samples.append(sample)
                _save_sample(handle, sample, origin_ns)

        for sequence in range(args.latency_pairs):
            for endpoint in (old, new) if sequence % 2 == 0 else (new, old):
                sample = _measure_one(
                    endpoint, schedule[sequence], "latency", 0, sequence, args.timeout
                )
                samples.append(sample)
                _save_sample(handle, sample, origin_ns)

        # Each arm has its own complete wave. Alternating wave order across
        # rounds avoids simultaneous old/new inference and reduces order bias.
        for round_number in range(args.rounds):
            endpoints = (old, new) if round_number % 2 == 0 else (new, old)
            for endpoint in endpoints:
                with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                    futures = [
                        pool.submit(
                            _measure_one,
                            endpoint,
                            schedule[sequence],
                            "throughput",
                            round_number,
                            sequence,
                            args.timeout,
                        )
                        for sequence in range(args.throughput_requests)
                    ]
                    wave = [future.result() for future in as_completed(futures)]
                for sample in sorted(wave, key=lambda item: item.sequence):
                    samples.append(sample)
                    _save_sample(handle, sample, origin_ns)

    summary = summarize(samples, old_meta, new_meta)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "SystemOne HTTP performance and response conformance; no eval-quality score",
        "model": args.model,
        "old": old_meta,
        "new": new_meta,
        "source_commit": _source_commit(),
        "harness_sha256": _harness_digest(),
        "fixture_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        "cohort_sha256": cohort_sha256(cases),
        "case_ids": [case.id for case in cases],
        "settings": settings,
        "summary": summary,
    }
    (args.output_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"receipt": str(args.output_dir / "receipt.json"), "summary": summary},
            sort_keys=True,
        )
    )
    return int(
        any(
            summary["arms"][arm][phase]["errors"]
            for arm in ("old", "new")
            for phase in ("warmup", "latency", "throughput")
        )
    )


def _matrix(args: argparse.Namespace) -> int:
    matrix = build_matrix(args.receipts)
    output = json.dumps(matrix, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(output)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(output)
        print(args.output)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run", help="measure one model against two SystemOne services"
    )
    run.add_argument("--model", required=True, choices=MODELS)
    run.add_argument("--old-url", required=True)
    run.add_argument("--new-url", required=True)
    run.add_argument("--old-token-env")
    run.add_argument("--new-token-env")
    for arm in ("old", "new"):
        run.add_argument(f"--{arm}-source-ref", required=True)
        run.add_argument(f"--{arm}-model-revision", required=True)
        run.add_argument(f"--{arm}-hardware", required=True)
        run.add_argument(
            f"--{arm}-network-scope",
            required=True,
            choices=("loopback", "private", "public"),
        )
    run.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    run.add_argument("--warmup", type=_nonnegative_int, default=4)
    run.add_argument("--latency-pairs", type=_positive_int, default=32)
    run.add_argument("--throughput-requests", type=_positive_int, default=64)
    run.add_argument("--rounds", type=_positive_int, default=2)
    run.add_argument("--concurrency", type=_positive_int, default=4)
    run.add_argument("--seed", type=int, default=17)
    run.add_argument("--timeout", type=_positive_float, default=60.0)
    run.add_argument("--output-dir", required=True, type=Path)
    run.set_defaults(handler=_run)

    matrix = commands.add_parser(
        "matrix", help="collate exactly six per-model receipts"
    )
    matrix.add_argument("--receipts", nargs="+", required=True, type=Path)
    matrix.add_argument("--output", type=Path)
    matrix.set_defaults(handler=_matrix)

    add_semantic_parsers(commands)

    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        # URL, HTTP body, and token values must never be printed from exceptions.
        print(
            f"decision-http-pair: {type(error).__name__}: {error if isinstance(error, ValueError) else 'operation failed'}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
