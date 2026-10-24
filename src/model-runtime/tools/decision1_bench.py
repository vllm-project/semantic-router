"""Latency and throughput of a Decision 1.0 package: the bundled runtime against the decision1 family.

    python3 tools/decision1_bench.py paired --package DIR --model REPO --cache-dir DIR \\
        --device cpu|rocm:0 --prompts NAME:REQUESTS.jsonl:COUNT --output OUT.json \\
        [--revision REV] [--threads N] [--router QUESTIONS.json]
    python3 tools/decision1_bench.py reference --package DIR --repo REPO --device cpu|cuda:0 \\
        --prompts NAME:REQUESTS.jsonl:COUNT --output OUT.json [--threads N] [--router QUESTIONS.json]
    python3 tools/decision1_bench.py native --model REPO --revision REV --cache-dir DIR \\
        --device cpu|rocm:0 --prompts NAME:REQUESTS.jsonl:COUNT --output OUT.json \\
        [--max-num-seqs N] [--max-num-batched-tokens N] [--enable-shared-context] \\
        [--concurrency 1 4 16] [--threads N] [--router QUESTIONS.json]

Single requests: every prompt once as one request, sequentially, after an
untimed warm-up pass (GPU graphs captured, caches filled); p50, p95 and mean
latency and the sequential rate. The bundled runtime is a library that answers
one request at a time, so its sequential rate is its throughput.

``paired`` loads both sides in one process and times every request on each
path back to back, rotating their order, so a shared machine's load changes
hit all paths alike: ``bundled`` (the package's ``system_one``), ``direct``
(the family's planning and ``run`` on the released physical batches: the same
work as ``bundled``), ``scheduler`` (planning and the scheduler with the engine
arguments, no API layer) and ``call`` (end to end through ``Runtime.call``). ``bundled`` and
``direct`` run on the thread the scheduler runs batches on, so every path
uses the same CPU thread pool. ``native`` alone also measures throughput through the
scheduler with C concurrent requests per wave. ``--router QUESTIONS.json``
replaces each prompt's questions with the router signals a Route-style
``QUESTIONS.json`` declares (explicit questions, so every Decision 1.0 model can
answer them). On a GPU both sides pin the built-in table's FLA kernel choices.
``--rounds N`` repeats a paired measurement (default 1, not an acceptance
verdict); ``--extend M`` continues through round M when evidence is under
five rounds or a latency interval is worse at the point but still Level.
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

from bench_stats import minus_baseline, should_extend, summary  # noqa: E402
from decision1_parity import panels, pin_status, require_fla_pin  # noqa: E402

ROUTER_SIGNALS = ("domain", "modality", "jailbreak", "safety", "pii", "fact_check")


def router_questions(path: str) -> dict[str, Any]:
    """Explicit System One questions for the router signals of a ``QUESTIONS.json``."""
    from vllm_srun.families.decision1.package import presets

    expanded = presets(Path(path).parent)
    return {name: expanded[name] for name in ROUTER_SIGNALS if name in expanded}


def requests(args: argparse.Namespace) -> list[dict[str, Any]]:
    questions = router_questions(args.router) if args.router else None
    return [
        {"state": prompt["state"], "questions": questions or prompt["questions"]}
        for _, prompt in panels(args.prompts)
    ]


def wire_size(body: dict[str, Any]) -> int:
    """The encoded body size the HTTP server passes to ``Runtime.call`` (small bodies plan inline)."""
    return len(json.dumps(body, separators=(",", ":")).encode("utf-8"))


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


def engine_args(args: argparse.Namespace) -> Any:
    """The served model's engine arguments: the defaults, with the ones given on the command line."""
    from vllm_srun.config import EngineArgs

    given = {
        name: getattr(args, name)
        for name in ("max_num_seqs", "max_num_batched_tokens", "enable_shared_context")
        if getattr(args, name, None) is not None
    }
    return EngineArgs(**given)


def reference_device(device: str) -> str:
    """The bundled runtime's name for a device (it predates ``rocm:N``)."""
    return device.replace("rocm", "cuda")


class Native:
    """The decision1 family behind a ``Runtime``, with each way of sending it a request."""

    def __init__(self, args: argparse.Namespace) -> None:
        from vllm_srun.config import ModelConfig, ServeConfig
        from vllm_srun.runtime import Runtime

        self.runtime = Runtime(
            ServeConfig(
                models=(
                    ModelConfig(
                        model=args.model, revision=args.revision, device=args.device
                    ),
                ),
                cache_dir=args.cache_dir,
                offline=True,
                threads=args.threads,
                engine_args=engine_args(args),
            )
        )
        self.runtime.start(background=False)
        self.served = self.runtime.lookup(None)
        self.loop = asyncio.new_event_loop()
        self.options = {"return_meta": False}

    def call(self, body: dict[str, Any]) -> None:
        """End to end through ``Runtime.call``, as the HTTP server calls it."""
        request = {**body, "options": self.options}
        status, _ = self.loop.run_until_complete(
            self.runtime.call("decisions", request, wire_size(request))
        )
        if status != 200:
            raise RuntimeError(f"HTTP {status}")

    def scheduler(self, body: dict[str, Any]) -> None:
        """Planning and the scheduler, without the API layer."""
        model = self.served.model
        plan = model.plan(body["state"], body["questions"])
        self.served.submit_items(plan.items, None).result()

    def direct(self, body: dict[str, Any]) -> None:
        """Planning and ``run`` on the released physical batches, on the calling thread."""
        model = self.served.model
        items = model.plan(body["state"], body["questions"]).items
        batches = model.exact_batches(items)
        if batches is None:
            batches = [list(range(len(items)))] if items else []
        for batch in batches:
            model.run([items[index] for index in batch])

    def on_device(
        self, call: Callable[[dict[str, Any]], None]
    ) -> Callable[[dict[str, Any]], None]:
        """``call`` on the thread the scheduler runs batches on (the CPU device thread).

        CPU thread pools are per calling thread: paths timed on different
        threads would keep two pools spinning on the same cores.
        """
        return lambda body: self.served.scheduler.execute(lambda: call(body))

    def throughput(self, bodies: list[dict[str, Any]], concurrency: int) -> float:
        """Requests per second in waves of ``concurrency`` concurrent calls."""

        async def wave(batch: list[dict[str, Any]]) -> None:
            requests = [{**body, "options": self.options} for body in batch]
            results = await asyncio.gather(
                *(
                    self.runtime.call("decisions", request, wire_size(request))
                    for request in requests
                )
            )
            if any(status != 200 for status, _ in results):
                raise RuntimeError("a concurrent request failed")

        started = time.perf_counter()
        for start in range(0, len(bodies), concurrency):
            self.loop.run_until_complete(wave(bodies[start : start + concurrency]))
        return len(bodies) / (time.perf_counter() - started)

    def receipt(self) -> dict[str, Any]:
        """What ran: kernels, fast-path pieces, graphs per layer stack and reduced copies."""
        return {"engine": self.served.model.engine_model.receipt()}

    def close(self) -> None:
        self.loop.close()
        self.runtime.stop()


def run_reference(args: argparse.Namespace) -> dict[str, Any]:
    pinned, pin_reason = pin_status(args.repo, args.device)
    require_fla_pin(pinned, pin_reason, args.device)
    call = bundled(args)
    bodies = requests(args)
    timed(call, bodies[: args.warmup])
    return {
        "side": "reference",
        "fla_pinned": pinned,
        "fla_pin_reason": pin_reason,
        "single": summary(timed(call, bodies)),
    }


def run_native(args: argparse.Namespace) -> dict[str, Any]:
    native = Native(args)
    try:
        bodies = requests(args)
        timed(native.call, bodies[: args.warmup])
        result = {
            "side": "native",
            "engine_args": native.served.args.describe(native.served.shared),
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


PAIRED_METRICS = ("p50_ms", "p95_ms", "mean_ms", "requests_per_second")


def paired_round(
    paths: dict[str, Callable[[dict[str, Any]], None]], bodies: list[dict[str, Any]]
) -> tuple[dict[str, dict[str, float]], dict[str, list[float]]]:
    seconds: dict[str, list[float]] = {name: [] for name in paths}
    names = list(paths)
    for index, body in enumerate(bodies):
        shift = index % len(names)
        for name in names[shift:] + names[:shift]:
            seconds[name] += timed(paths[name], [body])
    return {name: summary(values) for name, values in seconds.items()}, seconds


def run_paired(args: argparse.Namespace) -> dict[str, Any]:
    pinned, pin_reason = pin_status(args.model, args.device)
    require_fla_pin(pinned, pin_reason, args.device)
    native = Native(args)
    try:
        paths = {
            "bundled": native.on_device(bundled(args)),
            "direct": native.on_device(native.direct),
            "scheduler": native.scheduler,
            "call": native.call,
        }
        bodies = requests(args)
        for call in paths.values():
            timed(call, bodies[: args.warmup])
        names = list(paths)
        target = args.rounds
        rounds: list[dict[str, Any]] = []
        last_seconds: dict[str, list[float]] | None = None
        while True:
            summaries, last_seconds = paired_round(paths, bodies)
            summaries["round"] = len(rounds)
            rounds.append(summaries)
            if len(rounds) < target:
                continue
            intervals = {
                name: minus_baseline(
                    [row[name] for row in rounds],
                    [row["bundled"] for row in rounds],
                    PAIRED_METRICS,
                )
                for name in names[1:]
            }
            if args.extend > target and any(
                should_extend(intervals[name], ("p50_ms", "p95_ms", "mean_ms"))
                for name in intervals
            ):
                target = args.extend
                continue
            break
        assert last_seconds is not None
        mean_summary = {
            name: {
                metric: statistics.fmean(row[name][metric] for row in rounds)
                for metric in PAIRED_METRICS
            }
            | {"requests": last_seconds[name].__len__()}
            for name in names
        }
        result = {
            "side": "paired",
            "fla_pinned": pinned,
            "fla_pin_reason": pin_reason,
            "rounds": rounds,
            "intervals_minus_bundled": intervals,
            **mean_summary,
            "minus_bundled_ms": {
                name: paired_difference(last_seconds[name], last_seconds["bundled"])
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
        command.add_argument("--max-num-seqs", type=int)
        command.add_argument("--max-num-batched-tokens", type=int)
        command.add_argument(
            "--enable-shared-context",
            action=argparse.BooleanOptionalAction,
            default=None,
        )
    for command in (reference, native, paired):
        command.add_argument("--device", default="cpu")
        command.add_argument("--prompts", action="append", required=True)
        command.add_argument("--router")
        command.add_argument("--warmup", type=int, default=64)
        command.add_argument("--threads", type=int)
        command.add_argument("--output", required=True)
        command.add_argument("--rounds", type=int, default=1)
        command.add_argument(
            "--extend",
            type=int,
            default=1,
            help="max rounds if a latency interval is worse at the point but still Level",
        )
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
