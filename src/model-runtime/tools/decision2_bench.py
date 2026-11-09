"""Latency of a Decision 2.0 package: native engine against the released runtime.

    python3 tools/decision2_bench.py paired --package DIR --repo REPO --prompts NAME:REQUESTS.jsonl:COUNT \\
        --output OUT.json --device cuda:0 [--base-path DIR] [--warmup-passes N]
    python3 tools/decision2_bench.py sequential --package DIR --repo REPO --prompts NAME:REQUESTS.jsonl:COUNT \\
        --output OUT.json --device cuda:0 [--base-path DIR]

``paired`` loads both copies at once and times every request on each path
back to back, rotating order, with a GPU synchronize after each call.
``sequential`` loads one side, times the panel, frees it, then times the
other. Use sequential when both copies plus working memory may not fit
(Vega-27B on one H200). Sequential is not simultaneous-residency paired
timing; the receipt says so.

Both paths answer the same ``system_one`` work. After timing, every response
must contain the request's question IDs, nonempty answers, and no item-level
error codes. ``--warmup-passes`` (default 2) runs the full panel untimed so
CUDA graphs capture on a shape's second use before the clock starts; the
receipt records capture/replay/eager counts and refuses a timed panel that
still captured. ``--rounds N`` repeats the measurement (default 1, which is
not an acceptance verdict); ``--extend M`` continues through round M when
evidence is under five rounds or a latency or throughput interval is worse at the point
but still Level. Sequential alternates which side loads first each round.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from collections.abc import Callable
from contextlib import ExitStack, closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench_answers import validate_panel  # noqa: E402
from bench_stats import (  # noqa: E402
    minus_baseline,
    paired_difference,
    should_extend,
    summary,
)
from decision1_parity import panels, pin_status, require_fla_pin  # noqa: E402
from gpu_parity import load as load_native  # noqa: E402
from gpu_parity import system_one  # noqa: E402


def requests(args: argparse.Namespace) -> list[dict[str, Any]]:
    bodies = [
        {"id": prompt["id"], "state": prompt["state"], "questions": prompt["questions"]}
        for _, prompt in panels(args.prompts)
    ]
    if not bodies:
        raise SystemExit("the benchmark panel must contain at least one request")
    return bodies


def synchronize(device: str) -> None:
    if device == "cpu":
        return
    import torch

    torch.cuda.synchronize(device)


@dataclass
class BenchmarkSide:
    """One loaded model, with explicit ownership for sequential measurements."""

    model: Any
    device: str
    native: bool = False

    def __call__(self, body: dict[str, Any]) -> dict[str, Any]:
        if self.native:
            response = system_one(self.model, None, body["state"], body["questions"])
        else:
            response = self.model.system_one(
                state=body["state"], questions=body["questions"]
            )
        synchronize(self.device)
        return response["answers"]

    def graph_stats(self) -> dict[str, int]:
        """Read native graph counters; the released runtime has none here."""
        if not self.native:
            return {}
        receipt = self.model.engine_model.receipt()
        counters = receipt.get("graphs", receipt.get("encoder_graphs", {}))
        return {key: int(counters.get(key, 0)) for key in GRAPH_COUNTERS}

    def close(self) -> None:
        self.model = None
        gc.collect()
        if self.device != "cpu":
            import torch

            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()


GRAPH_COUNTERS = ("captures", "replays", "eager")


def graph_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {key: after.get(key, 0) - before.get(key, 0) for key in GRAPH_COUNTERS}


def refuse_capture_during_timing(delta: dict[str, int], side: str) -> None:
    if delta.get("captures", 0) > 0:
        raise SystemExit(
            f"{side}: {delta['captures']} CUDA graph captures during the timed panel"
        )


def warmup(call: Callable, bodies: list[dict[str, Any]], passes: int) -> None:
    for _ in range(passes):
        for body in bodies:
            call(body)


def bundled(args: argparse.Namespace) -> BenchmarkSide:
    from transformers import AutoModel

    options = {"trust_remote_code": True, "device": args.device.replace("rocm", "cuda")}
    if args.base_path:
        options["base_path"] = args.base_path
    model = AutoModel.from_pretrained(args.package, **options)
    return BenchmarkSide(model, options["device"])


def native(args: argparse.Namespace) -> BenchmarkSide:
    ns = argparse.Namespace(
        package=args.package,
        base_path=args.base_path,
        device=args.device,
        no_graphs=args.no_graphs,
        no_fused=args.no_fused,
    )
    model = load_native(ns)
    return BenchmarkSide(model, args.device.replace("rocm", "cuda"), native=True)


def time_answers(
    call: Callable[[dict[str, Any]], dict[str, Any]], bodies: list[dict[str, Any]]
) -> tuple[list[float], list[Any]]:
    seconds = []
    collected = []
    for body in bodies:
        started = time.perf_counter()
        answers = call(body)
        seconds.append(time.perf_counter() - started)
        collected.append(answers)
    return seconds, collected


D2_METRICS = ("p50_ms", "p95_ms", "mean_ms", "requests_per_second")


def _mean_summaries(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: statistics.fmean(row[key] for row in rows) for key in D2_METRICS} | {
        "requests": rows[0]["requests"]
    }


def run_paired(args: argparse.Namespace) -> dict[str, Any]:
    pinned, pin_reason = pin_status(args.repo, args.device)
    require_fla_pin(pinned, pin_reason, args.device)
    bodies = requests(args)
    with ExitStack() as stack:
        released = stack.enter_context(closing(bundled(args)))
        engine = stack.enter_context(closing(native(args)))
        result = paired_measurements(args, bodies, released, engine)
    return {**result, "fla_pinned": pinned, "fla_pin_reason": pin_reason}


def paired_measurements(
    args: argparse.Namespace,
    bodies: list[dict[str, Any]],
    released: BenchmarkSide,
    engine: BenchmarkSide,
) -> dict[str, Any]:
    warmup(released, bodies, args.warmup_passes)
    warmup(engine, bodies, args.warmup_passes)
    names = ("released", "native")
    paths = {"released": released, "native": engine}
    first_released = first_native = None
    target = args.rounds
    rounds: list[dict[str, Any]] = []
    last_seconds: dict[str, list[float]] = {"released": [], "native": []}
    graphs_before = engine.graph_stats()
    graphs_timed: list[dict[str, int]] = []
    while True:
        seconds = {"released": [], "native": []}
        collected = {"released": [], "native": []}
        before = engine.graph_stats()
        for index, body in enumerate(bodies):
            shift = (index + len(rounds)) % 2
            order = names[shift:] + names[:shift]
            for name in order:
                started = time.perf_counter()
                answers = paths[name](body)
                seconds[name].append(time.perf_counter() - started)
                collected[name].append(answers)
        delta = graph_delta(before, engine.graph_stats())
        refuse_capture_during_timing(delta, "native")
        graphs_timed.append(delta)
        validate_panel(collected["released"], bodies, "released")
        validate_panel(collected["native"], bodies, "native")
        if first_released is None:
            first_released, first_native = (
                collected["released"][0],
                collected["native"][0],
            )
        last_seconds = seconds
        rounds.append(
            {
                "round": len(rounds),
                "released": summary(seconds["released"]),
                "native": summary(seconds["native"]),
                "graphs": delta,
            }
        )
        if len(rounds) < target:
            continue
        intervals = minus_baseline(
            [row["native"] for row in rounds],
            [row["released"] for row in rounds],
            D2_METRICS,
        )
        if args.extend > target and should_extend(intervals, D2_METRICS):
            target = args.extend
            continue
        break
    return {
        "side": "paired",
        "methodology": "simultaneous-residency",
        "warmup_passes": args.warmup_passes,
        "graphs_after_warmup": graphs_before,
        "graphs_timed": graphs_timed,
        "rounds": rounds,
        "intervals_minus_released": intervals,
        "released": _mean_summaries([row["released"] for row in rounds]),
        "native": _mean_summaries([row["native"] for row in rounds]),
        "minus_released_ms": paired_difference(
            last_seconds["native"], last_seconds["released"]
        ),
        "first_answers_identical": json.dumps(first_released, sort_keys=True)
        == json.dumps(first_native, sort_keys=True),
    }


def _load_and_time(
    kind: str, args: argparse.Namespace, bodies: list[dict[str, Any]]
) -> tuple[list[float], Any, dict[str, dict[str, int]]]:
    load = bundled if kind == "released" else native
    with closing(load(args)) as side:
        warmup(side, bodies, args.warmup_passes)
        before = side.graph_stats()
        seconds, collected = time_answers(side, bodies)
        delta = graph_delta(before, side.graph_stats())
        refuse_capture_during_timing(delta, kind)
        validate_panel(collected, bodies, kind)
        stats = {"graphs_after_warmup": before, "graphs_timed": delta}
        return seconds, collected[0], stats


def run_sequential(args: argparse.Namespace) -> dict[str, Any]:
    pinned, pin_reason = pin_status(args.repo, args.device)
    require_fla_pin(pinned, pin_reason, args.device)
    bodies = requests(args)
    target = args.rounds
    rounds: list[dict[str, Any]] = []
    first_released = first_native = None
    last_released: list[float] = []
    last_native: list[float] = []
    while True:
        order = (
            ("released", "native") if len(rounds) % 2 == 0 else ("native", "released")
        )
        seconds: dict[str, list[float]] = {}
        firsts: dict[str, Any] = {}
        stats: dict[str, Any] = {}
        for kind in order:
            seconds[kind], firsts[kind], stats[kind] = _load_and_time(
                kind, args, bodies
            )
        if first_released is None:
            first_released, first_native = firsts["released"], firsts["native"]
        last_released, last_native = seconds["released"], seconds["native"]
        rounds.append(
            {
                "round": len(rounds),
                "order": list(order),
                "released": summary(seconds["released"]),
                "native": summary(seconds["native"]),
                "graphs": stats,
            }
        )
        if len(rounds) < target:
            continue
        intervals = minus_baseline(
            [row["native"] for row in rounds],
            [row["released"] for row in rounds],
            D2_METRICS,
        )
        if args.extend > target and should_extend(intervals, D2_METRICS):
            target = args.extend
            continue
        break
    return {
        "side": "sequential",
        "methodology": "alternating-loads",
        "fla_pinned": pinned,
        "fla_pin_reason": pin_reason,
        "warmup_passes": args.warmup_passes,
        "note": (
            "Each round loads one side, warms the full panel, times it, frees it, "
            "then the other. Load order alternates. This is not simultaneous-"
            "residency paired timing."
        ),
        "rounds": rounds,
        "intervals_minus_released": intervals,
        "released": _mean_summaries([row["released"] for row in rounds]),
        "native": _mean_summaries([row["native"] for row in rounds]),
        "minus_released_ms": paired_difference(last_native, last_released),
        "first_answers_identical": json.dumps(first_released, sort_keys=True)
        == json.dumps(first_native, sort_keys=True),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("paired", "sequential"):
        command = commands.add_parser(name)
        command.add_argument("--package", required=True)
        command.add_argument("--base-path")
        command.add_argument("--repo", help="built-in id for FLA pins")
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--prompts", action="append", required=True)
        command.add_argument("--warmup", type=int, help=argparse.SUPPRESS)
        command.add_argument(
            "--warmup-passes",
            type=int,
            default=2,
            help="untimed full-panel passes so every measured shape is captured before timing",
        )
        command.add_argument("--no-graphs", action="store_true")
        command.add_argument("--no-fused", action="store_true")
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--rounds", type=int, default=1)
        command.add_argument("--extend", type=int, default=1)
    args = parser.parse_args()
    if args.warmup is not None:
        parser.error("--warmup was replaced by --warmup-passes (full-panel passes)")
    if args.rounds < 1 or args.extend < 1:
        parser.error("--rounds and --extend must be positive")
    if args.warmup_passes < 0:
        parser.error("--warmup-passes must be nonnegative")
    if args.device != "cpu" and not args.repo:
        parser.error("--repo is required on a GPU to check the recorded FLA choices")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    started = time.time()
    run = {"paired": run_paired, "sequential": run_sequential}[args.command]
    result = run(args)
    result.update(
        device=args.device,
        prompts=args.prompts,
        started_unix=started,
        schema="model-runtime-decision2-bench/1",
    )
    args.output.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"side": result["side"], "methodology": result["methodology"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
