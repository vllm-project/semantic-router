"""Latency and throughput of a Decision 1.0 package: the bundled runtime against the decision1 family.

    python3 tools/decision1_bench.py paired --package DIR --model REPO --cache-dir DIR \\
        --device cpu|rocm:0 --prompts NAME:REQUESTS.jsonl:COUNT --output OUT.json \\
        [--revision REV] [--threads N] [--router QUESTIONS.json]
    python3 tools/decision1_bench.py reference --package DIR --repo REPO --device cpu|cuda:0 \\
        --prompts NAME:REQUESTS.jsonl:COUNT --output OUT.json [--threads N] [--router QUESTIONS.json]
    python3 tools/decision1_bench.py native --model REPO --revision REV --cache-dir DIR \\
        --device cpu|rocm:0 --prompts NAME:REQUESTS.jsonl:COUNT --output OUT.json \\
        [--profile P] [--concurrency 1 4 16] [--threads N] [--router QUESTIONS.json]

Single requests: every prompt once as one request, sequentially, after an
untimed warm-up pass (GPU graphs captured, caches filled); p50, p95 and mean
latency and the sequential rate. The bundled runtime is a library that answers
one request at a time, so its sequential rate is its throughput.

``paired`` loads both sides in one process and times every request on each
path back to back, rotating their order, so a shared machine's load changes
hit all paths alike: ``bundled`` (the package's ``system_one``), ``direct``
(the family's planning and ``run`` on the exact profile's physical batches, on
the calling thread: the same work as ``bundled``), ``scheduler`` (planning and
the scheduler, no API layer) and ``call`` (end to end through
``Runtime.call``). ``native`` alone also measures throughput through the
scheduler with C concurrent requests per wave. ``--router QUESTIONS.json``
replaces each prompt's questions with the router signals a Route-style
``QUESTIONS.json`` declares (explicit questions, so every Decision 1.0 model can
answer them). On a GPU both sides pin the built-in table's FLA kernel choices.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from collections.abc import Callable
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


def bundled(args: argparse.Namespace) -> Callable[[dict[str, Any]], None]:
    """The package's own runtime answering one request (synchronized on a GPU)."""
    import torch

    if args.threads:
        torch.set_num_threads(args.threads)
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        args.package, trust_remote_code=True, device=reference_device(args.device)
    )

    def call(body: dict[str, Any]) -> None:
        model.system_one(state=body["state"], questions=body["questions"])
        if args.device != "cpu":
            torch.cuda.synchronize()

    return call


def reference_device(device: str) -> str:
    """The bundled runtime's name for a device (it predates ``rocm:N``)."""
    return device.replace("rocm", "cuda")


class Native:
    """The decision1 family behind a ``Runtime``, with each way of sending it a request."""

    def __init__(self, args: argparse.Namespace) -> None:
        from vllm_sr_runtime.config import ServeConfig
        from vllm_sr_runtime.runtime import Runtime

        self.runtime = Runtime(
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
        self.runtime.start(background=False)
        self.served = self.runtime.primary
        self.profile = args.profile
        self.loop = asyncio.new_event_loop()
        self.options = {"return_meta": False, "profile": args.profile}

    def call(self, body: dict[str, Any]) -> None:
        """End to end through ``Runtime.call``."""
        status, _ = self.loop.run_until_complete(
            self.runtime.call("decisions", {**body, "options": self.options})
        )
        if status != 200:
            raise RuntimeError(f"HTTP {status}")

    def scheduler(self, body: dict[str, Any]) -> None:
        """Planning and the scheduler, without the API layer."""
        model = self.served.model
        plan = model.plan(body["state"], body["questions"])
        self.served.submit_items(plan.items, None, self.profile).result()

    def direct(self, body: dict[str, Any]) -> None:
        """Planning and ``run`` on the exact physical batches, on the calling thread."""
        model = self.served.model
        items = model.plan(body["state"], body["questions"]).items
        for batch in model.exact_batches(items) or [list(range(len(items)))]:
            model.run([items[index] for index in batch])

    def throughput(self, bodies: list[dict[str, Any]], concurrency: int) -> float:
        """Requests per second in waves of ``concurrency`` concurrent calls."""

        async def wave(batch: list[dict[str, Any]]) -> None:
            results = await asyncio.gather(
                *(
                    self.runtime.call("decisions", {**body, "options": self.options})
                    for body in batch
                )
            )
            if any(status != 200 for status, _ in results):
                raise RuntimeError("a concurrent request failed")

        started = time.perf_counter()
        for start in range(0, len(bodies), concurrency):
            self.loop.run_until_complete(wave(bodies[start : start + concurrency]))
        return len(bodies) / (time.perf_counter() - started)

    def receipt(self) -> dict[str, Any]:
        engine = self.served.model.engine_model
        out = {"fast_path": getattr(engine, "fast", None)}
        graphs = getattr(engine, "graphs", None)
        if graphs is not None:
            out["graphs"] = graphs.receipt()
        return out

    def close(self) -> None:
        self.loop.close()
        self.runtime.stop()


def run_reference(args: argparse.Namespace) -> dict[str, Any]:
    pinned = pin(args.repo, args.device)
    call = bundled(args)
    bodies = requests(args)
    timed(call, bodies[: args.warmup])
    return {
        "side": "reference",
        "fla_pinned": pinned,
        "single": summary(timed(call, bodies)),
    }


def run_native(args: argparse.Namespace) -> dict[str, Any]:
    native = Native(args)
    try:
        bodies = requests(args)
        timed(native.call, bodies[: args.warmup])
        result = {
            "side": "native",
            "profile": args.profile,
            "single": summary(timed(native.call, bodies)),
            "single_scheduler": summary(timed(native.scheduler, bodies)),
            "throughput": {
                str(concurrency): native.throughput(bodies, concurrency)
                for concurrency in args.concurrency
            },
        }
        return {**result, **native.receipt()}
    finally:
        native.close()


def run_paired(args: argparse.Namespace) -> dict[str, Any]:
    pinned = pin(args.model, args.device)
    native = Native(args)
    try:
        paths = {
            "bundled": bundled(args),
            "direct": native.direct,
            "scheduler": native.scheduler,
            "call": native.call,
        }
        bodies = requests(args)
        for call in paths.values():
            timed(call, bodies[: args.warmup])
        seconds: dict[str, list[float]] = {name: [] for name in paths}
        names = list(paths)
        for index, body in enumerate(bodies):
            shift = index % len(names)
            for name in names[shift:] + names[:shift]:
                seconds[name] += timed(paths[name], [body])
        result = {
            "side": "paired",
            "fla_pinned": pinned,
            **{name: summary(values) for name, values in seconds.items()},
            "minus_bundled_ms": {
                name: paired_difference(seconds[name], seconds["bundled"])
                for name in names[1:]
            },
        }
        return {**result, **native.receipt()}
    finally:
        native.close()


def paired_difference(path: list[float], base: list[float]) -> dict[str, float]:
    """Median and mean of the per-request differences ``path - base``, in ms."""
    differences = [1000 * (a - b) for a, b in zip(path, base, strict=True)]
    return {
        "p50": statistics.median(differences),
        "mean": statistics.fmean(differences),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    reference = commands.add_parser("reference")
    reference.add_argument("--repo")
    native = commands.add_parser("native")
    native.add_argument("--concurrency", type=int, nargs="*", default=[1, 4, 16])
    paired = commands.add_parser("paired")
    for command in (reference, paired):
        command.add_argument("--package", required=True)
    for command in (native, paired):
        command.add_argument("--model", required=True)
        command.add_argument("--revision")
        command.add_argument("--cache-dir")
        command.add_argument("--profile", default="exact")
    for command in (reference, native, paired):
        command.add_argument("--device", default="cpu")
        command.add_argument("--prompts", action="append", required=True)
        command.add_argument("--router")
        command.add_argument("--warmup", type=int, default=64)
        command.add_argument("--threads", type=int)
        command.add_argument("--output", required=True)
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    started = time.time()
    run = {"reference": run_reference, "native": run_native, "paired": run_paired}
    result = run[args.command](args)
    result.update(
        device=args.device,
        threads=args.threads,
        prompts=args.prompts,
        router=bool(args.router),
        started_unix=started,
    )
    Path(args.output).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    shown = ("side", "single", "bundled", "direct", "scheduler", "call")
    print(json.dumps({key: result[key] for key in shown if key in result}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
