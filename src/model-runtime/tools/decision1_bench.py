"""Latency and throughput of a Decision 1.0 package: the bundled runtime against the decision1 family.

    python3 tools/decision1_bench.py reference --package DIR --repo REPO --device cpu|cuda:0 \\
        --prompts NAME:REQUESTS.jsonl:COUNT --output OUT.json [--threads N] [--router QUESTIONS.json]
    python3 tools/decision1_bench.py native --model REPO --revision REV --cache-dir DIR \\
        --device cpu|rocm:0 --prompts NAME:REQUESTS.jsonl:COUNT --output OUT.json \\
        [--profile P] [--concurrency 1 4 16] [--threads N] [--router QUESTIONS.json]

Single requests: every prompt once as one request, sequentially, after an
untimed warm-up pass (GPU graphs captured, caches filled); p50, p95 and mean
latency and the sequential rate. The bundled runtime is a library that answers
one request at a time, so its sequential rate is its throughput. ``native``
measures the request end to end through ``Runtime.call`` (``single``) and
through the model's planning and scheduler alone (``single_scheduler``, no API
layer), and throughput through the scheduler with C concurrent requests per
wave. ``--router QUESTIONS.json`` replaces each prompt's questions with the
router signals a Route-style ``QUESTIONS.json`` declares (explicit questions,
so every Decision 1.0 model can answer them). On a GPU both sides pin the
built-in table's FLA kernel choices.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from decision1_parity import panels, pin  # noqa: E402

ROUTER_SIGNALS = ("domain", "modality", "jailbreak", "safety", "pii", "fact_check")


def router_questions(path: str) -> dict[str, Any]:
    """Explicit System One questions for the router signals of a ``QUESTIONS.json``."""
    from vllm_sr_runtime.families.decision1.package import presets

    expanded = presets(Path(path).parent)
    return {name: expanded[name] for name in ROUTER_SIGNALS if name in expanded}


def requests(args: argparse.Namespace) -> list[dict[str, Any]]:
    questions = router_questions(args.router) if args.router else None
    return [
        {"state": prompt["state"], "questions": questions or prompt["questions"]}
        for _, prompt in panels(args.prompts)
    ]


def summary(seconds: list[float]) -> dict[str, float]:
    ordered = sorted(seconds)
    return {
        "requests": len(ordered),
        "p50_ms": 1000 * statistics.median(ordered),
        "p95_ms": 1000
        * ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))],
        "mean_ms": 1000 * statistics.fmean(ordered),
        "requests_per_second": len(ordered) / sum(ordered),
    }


def timed(call, bodies: list[dict[str, Any]]) -> list[float]:
    seconds = []
    for body in bodies:
        started = time.perf_counter()
        call(body)
        seconds.append(time.perf_counter() - started)
    return seconds


def run_reference(args: argparse.Namespace) -> dict[str, Any]:
    pinned = pin(args.repo, args.device)
    import torch

    if args.threads:
        torch.set_num_threads(args.threads)
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        args.package, trust_remote_code=True, device=args.device
    )

    def call(body: dict[str, Any]) -> None:
        model.system_one(state=body["state"], questions=body["questions"])
        if args.device != "cpu":
            torch.cuda.synchronize()

    bodies = requests(args)
    timed(call, bodies[: args.warmup])
    return {
        "side": "reference",
        "fla_pinned": pinned,
        "single": summary(timed(call, bodies)),
    }


def run_native(args: argparse.Namespace) -> dict[str, Any]:
    from vllm_sr_runtime.config import ServeConfig
    from vllm_sr_runtime.runtime import Runtime

    runtime = Runtime(
        ServeConfig(
            model=args.model,
            revision=args.revision,
            device=args.device,
            cache_dir=args.cache_dir,
            offline=True,
            profile=args.profile,
            threads=args.threads,
        )
    )
    runtime.start(background=False)
    loop = asyncio.new_event_loop()
    options = {"return_meta": False, "profile": args.profile}

    def call(body: dict[str, Any]) -> None:
        status, _ = loop.run_until_complete(
            runtime.call("decisions", {**body, "options": options})
        )
        if status != 200:
            raise RuntimeError(f"HTTP {status}")

    async def wave(batch: list[dict[str, Any]]) -> None:
        results = await asyncio.gather(
            *(runtime.call("decisions", {**body, "options": options}) for body in batch)
        )
        if any(status != 200 for status, _ in results):
            raise RuntimeError("a concurrent request failed")

    served = runtime.primary

    def scheduled(body: dict[str, Any]) -> None:
        plan = served.model.plan(body["state"], body["questions"])
        served.submit_items(plan.items, None, args.profile).result()

    try:
        bodies = requests(args)
        timed(call, bodies[: args.warmup])
        result = {
            "side": "native",
            "profile": args.profile,
            "fast_path": getattr(served.model.engine_model, "fast", None),
            "single": summary(timed(call, bodies)),
            "single_scheduler": summary(timed(scheduled, bodies)),
            "throughput": {},
        }
        for concurrency in args.concurrency:
            started = time.perf_counter()
            for start in range(0, len(bodies), concurrency):
                loop.run_until_complete(wave(bodies[start : start + concurrency]))
            result["throughput"][str(concurrency)] = len(bodies) / (
                time.perf_counter() - started
            )
        graphs = getattr(runtime.primary.model.engine_model, "graphs", None)
        if graphs is not None:
            result["graphs"] = graphs.receipt()
        return result
    finally:
        loop.close()
        runtime.stop()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    reference = commands.add_parser("reference")
    reference.add_argument("--package", required=True)
    reference.add_argument("--repo")
    native = commands.add_parser("native")
    native.add_argument("--model", required=True)
    native.add_argument("--revision")
    native.add_argument("--cache-dir")
    native.add_argument("--profile", default="exact")
    native.add_argument("--concurrency", type=int, nargs="*", default=[1, 4, 16])
    for command in (reference, native):
        command.add_argument("--device", default="cpu")
        command.add_argument("--prompts", action="append", required=True)
        command.add_argument("--router")
        command.add_argument("--warmup", type=int, default=64)
        command.add_argument("--threads", type=int)
        command.add_argument("--output", required=True)
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    started = time.time()
    result = run_reference(args) if args.command == "reference" else run_native(args)
    result.update(
        device=args.device,
        threads=args.threads,
        prompts=args.prompts,
        router=bool(args.router),
        started_unix=started,
    )
    Path(args.output).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("side", "single")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
