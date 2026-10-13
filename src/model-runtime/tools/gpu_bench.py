"""Latency and throughput of the native engine on one device (performance records).

    python3 tools/gpu_bench.py --package DIR --prompts PROMPTS.jsonl --output OUT.json [--count 400]
        [--device rocm:0] [--base-path DIR] [--concurrency 1,4,16,64] [--throughput-count 512]

latency     the first --count prompts as single requests on the exact profile, --warmup-passes untimed passes
            (graphs are captured on a shape's second use) then one timed pass: p50 / p95 / mean, requests/s
many        the public many-question request (``many_questions.py``) at 16, 64 and 128 questions as one request
            on the exact and the shared_context profiles: p50 / p95 of --runs requests
throughput  the first --throughput-count prompts in waves of C concurrent requests through the scheduler, for
            the exact and the batching profiles, two untimed passes then a timed one: requests/s, the mean wave
            latency and the timed pass's graph captures / replays / eager forwards per C
Every request is timed with the device synchronized around it.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import many_questions
from gpu_parity import load, system_one
from vllm_srun.profiles.batching import BatchingProfile
from vllm_srun.profiles.exact import ExactProfile
from vllm_srun.profiles.shared_context import SharedContextProfile
from vllm_srun.scheduler.scheduler import Scheduler, SchedulerLimits


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def summary(milliseconds: list[float]) -> dict[str, float]:
    return {
        "p50": percentile(milliseconds, 0.5),
        "p95": percentile(milliseconds, 0.95),
        "mean": statistics.mean(milliseconds),
        "min": min(milliseconds),
        "max": max(milliseconds),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--package", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=400)
    parser.add_argument("--warmup-passes", type=int, default=2)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--concurrency", default="1,4,16,64")
    parser.add_argument("--throughput-count", type=int, default=512)
    parser.add_argument("--device", default="rocm:0")
    parser.add_argument("--base-path")
    parser.add_argument("--no-graphs", action="store_true")
    parser.add_argument("--no-fused", action="store_true")
    args = parser.parse_args()
    import torch

    model = load(args)
    exact = ExactProfile()
    prompts = [json.loads(line) for line in open(args.prompts, encoding="utf-8")]
    result: dict = {
        "schema": "model-runtime-gpu-bench/1",
        "model": model.info.id,
        "device": model.engine_model.device_info.name,
        "fast_path_options": {
            "graphs": not args.no_graphs,
            "fused_kernels": not args.no_fused,
        },
    }

    def timed(state, questions, profile=exact):
        torch.cuda.synchronize()
        started = time.perf_counter()
        response = system_one(model, profile, state, questions)
        torch.cuda.synchronize()
        return 1000 * (time.perf_counter() - started), response

    single = prompts[: args.count]
    for _ in range(args.warmup_passes):
        for prompt in single:
            system_one(model, exact, prompt["state"], prompt["questions"])
    milliseconds = [timed(p["state"], p["questions"])[0] for p in single]
    result["latency"] = {
        "requests": len(single),
        "ms": summary(milliseconds),
        "requests_per_s": 1000 * len(single) / sum(milliseconds),
    }
    print(json.dumps({"latency": result["latency"]}), flush=True)

    request = many_questions.request(128)
    shared = SharedContextProfile()
    if shared.available(model):
        shared = None
    else:
        shared.bind(model)
    result["many"] = {}
    for n in (16, 64, 128):
        questions = dict(list(request["questions"].items())[:n])
        entry = {}
        for name, profile in (("exact", exact), ("shared_context", shared)):
            if profile is None:
                continue
            for _ in range(5):
                system_one(model, profile, request["state"], questions)
            milliseconds = [
                timed(request["state"], questions, profile)[0] for _ in range(args.runs)
            ]
            entry[name] = {
                "ms": summary(milliseconds),
                "shared_prefix": (
                    profile.share(
                        model.plan(request["state"], questions).items,
                        model.forward_token_budget(),
                    )
                    if name == "shared_context"
                    else 0
                ),
            }
        result["many"][str(n)] = {"ms": entry["exact"]["ms"], **entry}
        print(
            json.dumps(
                {"many": n, **{k: round(v["ms"]["p50"], 2) for k, v in entry.items()}}
            ),
            flush=True,
        )

    load_prompts = prompts[: args.throughput_count]
    plans = [model.plan(p["state"], p["questions"]) for p in load_prompts]
    result["throughput"] = {}
    for name, profile in (("exact", exact), ("batching", BatchingProfile())):
        scheduler = Scheduler(
            model, {name: profile}, SchedulerLimits(max_queue=4096, batch_window_ms=2.0)
        )
        scheduler.start()
        per = {}
        graphs = getattr(model.engine_model, "graphs", None)
        for concurrency in [int(c) for c in args.concurrency.split(",")]:
            for _ in range(3):
                before = dict(graphs.receipt()) if graphs is not None else {}
                latencies = []
                torch.cuda.synchronize()
                started = time.perf_counter()
                for s in range(0, len(plans), concurrency):
                    wave = plans[s : s + concurrency]
                    t = time.perf_counter()
                    futures = [
                        scheduler.submit(p.items, deadline=None, profile=name)
                        for p in wave
                    ]
                    for future in futures:
                        future.result()
                    latencies.append(1000 * (time.perf_counter() - t))
                torch.cuda.synchronize()
                seconds = time.perf_counter() - started
            per[str(concurrency)] = {
                "requests_per_s": len(plans) / seconds,
                "wave_ms_mean": statistics.mean(latencies),
            }
            if graphs is not None:
                after = graphs.receipt()
                per[str(concurrency)]["graphs"] = {
                    key: after[key] - before.get(key, 0)
                    for key in ("captures", "replays", "eager")
                }
            print(
                json.dumps(
                    {
                        "throughput": name,
                        "concurrency": concurrency,
                        **per[str(concurrency)],
                    }
                ),
                flush=True,
            )
        scheduler.stop()
        result["throughput"][name] = per
    result["fast_path"] = (
        model.engine_model.receipt() if hasattr(model.engine_model, "receipt") else None
    )
    with args.output.open("x", encoding="utf-8") as sink:
        json.dump(result, sink, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
