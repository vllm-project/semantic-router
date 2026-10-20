#!/usr/bin/env python3
"""Measure a running standalone Router against the deterministic provider fixture."""

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener


def request(endpoint, index):
    """Time a fresh HTTP connection, including the controlled backend response."""
    expected = "code-model" if index % 2 == 0 else "general-model"
    prompt = (
        "write code for binary search" if index % 2 == 0 else "Explain photosynthesis"
    )
    payload = {
        "model": "vllm-sr/auto",
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
    }
    call = Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with build_opener(ProxyHandler({})).open(call, timeout=10) as response:
        if response.headers.get_content_type() != "application/json":
            raise ValueError("expected a JSON response")
        body = response.read(1024 * 1024 + 1)
        if len(body) > 1024 * 1024:
            raise ValueError("response exceeds fixture size limit")
        result = json.loads(body)
    if not isinstance(result, dict) or result.get("model") != expected:
        raise ValueError("Router selected the wrong model")
    choices = result.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("response has no valid choices")
    message = choices[0].get("message")
    if not isinstance(message, dict) or message.get("content") != "ok":
        raise ValueError("fixture content was not preserved")
    return (time.perf_counter() - started) * 1000


def phase(endpoint, concurrency, seconds):
    if concurrency < 1 or seconds <= 0 or not math.isfinite(seconds):
        raise ValueError("concurrency and duration must be positive and finite")
    started = time.perf_counter()
    deadline = started + seconds

    def worker(worker_id):
        samples, errors = [], []
        routes = {"code-model": 0, "general-model": 0}
        index = worker_id
        while time.perf_counter() < deadline:
            try:
                samples.append(request(endpoint, index))
                routes["code-model" if index % 2 == 0 else "general-model"] += 1
            except (
                OSError,
                HTTPError,
                URLError,
                ValueError,
                KeyError,
                IndexError,
            ) as error:
                errors.append(type(error).__name__)
                break
            index += 1
        return samples, errors, routes

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(worker, range(concurrency)))
    elapsed = time.perf_counter() - started
    samples = sorted(sample for values, _, _ in results for sample in values)
    errors = [error for _, values, _ in results for error in values]

    def percentile(fraction):
        return samples[math.ceil(len(samples) * fraction) - 1] if samples else None

    return {
        "concurrency": concurrency,
        "seconds": elapsed,
        "requests": len(samples),
        "errors": errors,
        "route_requests": {
            model: sum(routes[model] for _, _, routes in results)
            for model in ("code-model", "general-model")
        },
        "requests_per_second": len(samples) / elapsed,
        "p50_ms": percentile(0.5),
        "p95_ms": percentile(0.95),
        "latencies_ms": samples,
        "passed": bool(samples) and not errors,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.seconds <= 0 or not math.isfinite(args.seconds) or args.repeats < 1:
        parser.error("duration and repeats must be positive and finite")
    if args.output.exists():
        parser.error("output already exists; choose a fresh path")
    report = {
        "scope": "HTTP client + standalone Router + deterministic provider; fresh connections",
        "warmup": {"latencies_ms": [], "errors": []},
        "phases": [],
        "passed": False,
    }
    try:
        for index in range(20):
            try:
                report["warmup"]["latencies_ms"].append(request(args.endpoint, index))
            except (OSError, ValueError, KeyError, IndexError) as error:
                report["warmup"]["errors"].append(type(error).__name__)
                return 1
        for repeat in range(1, args.repeats + 1):
            for concurrency in (1, 4, 8):
                result = {
                    "repeat": repeat,
                    **phase(args.endpoint, concurrency, args.seconds),
                }
                report["phases"].append(result)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                if not result["passed"]:
                    return 1
        report["passed"] = True
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
