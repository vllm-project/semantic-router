"""Latency and throughput of the embedding and rerank models through the task_heads family (embed workstream).

    python3 tools/embed_bench.py --package DIR --engine native|onnxruntime --output OUT.json
        [--device cpu|rocm:N] [--threads N] [--iterations N] [--warmup N] [--option KEY=JSON ...]

Loads the package through ``TaskHeadsFamily`` and an engine, and times the
model's ``run`` (the shared forward plus every head's readout: the native
engine's packed forward, or the onnxruntime exit graph in length buckets) on
synthetic token rows, so every engine sees identical inputs:

* embeddings: one text of 16, 64, 256 and 1,024 tokens per request (latency
  p50 / p95), and a request of 32 texts of 16-256 tokens (texts/s);
* rerank: one query of 12 tokens with 10 and 50 documents of 64-192 tokens
  (all pairs of a request in one forward; pairs/s).

``--option`` sets a model option (for example ``layers=[22]`` or
``pair_scorer={"layer":22,"dimension":768}``).
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from embed_corpus import embed_scenarios, rerank_scenarios  # noqa: E402
from vllm_srun.accel.cpu import CPUAccelerator  # noqa: E402
from vllm_srun.accel.rocm import ROCmAccelerator  # noqa: E402
from vllm_srun.engines.native.engine import NativeEngine  # noqa: E402
from vllm_srun.engines.onnxruntime.engine import OnnxRuntimeEngine  # noqa: E402
from vllm_srun.families.task_heads.family import TaskHeadsFamily  # noqa: E402
from vllm_srun.heads.task import Item  # noqa: E402
from vllm_srun.plugins.base import (  # noqa: E402
    EngineOptions,
    PackageRef,
    RegistryOptions,
)


def load(args: argparse.Namespace) -> Any:
    options = {
        key: json.loads(value) for key, value in (o.split("=", 1) for o in args.option)
    }
    family = TaskHeadsFamily(RegistryOptions(model_options=options))
    package = family.verify(PackageRef(args.package))
    spec = family.describe(package)
    if args.device == "cpu":
        accelerator: Any = CPUAccelerator()
        device = accelerator.devices()[0]
    else:
        accelerator = ROCmAccelerator()
        device = accelerator.devices()[int(args.device.split(":")[1])]
    engine = NativeEngine() if args.engine == "native" else OnnxRuntimeEngine()
    reason = engine.supports(spec, device)
    if reason:
        raise SystemExit(reason)
    model = family.load(
        package,
        spec,
        engine.load(spec, accelerator, device, EngineOptions(threads=args.threads)),
    )
    return model, int(package.details["package"].config["vocab_size"])


def timed(
    model: Any, items: list[Item], warmup: int, iterations: int
) -> dict[str, float]:
    for _ in range(warmup):
        model.run(items)
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        model.run(items)
        samples.append((time.perf_counter() - started) * 1000)
    samples.sort()
    return {
        "p50_ms": round(statistics.median(samples), 3),
        "p95_ms": round(samples[min(len(samples) - 1, int(0.95 * len(samples)))], 3),
        "mean_ms": round(statistics.fmean(samples), 3),
        "iterations": iterations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--engine", choices=("native", "onnxruntime"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--option", action="append", default=[])
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    model, vocab = load(args)
    planner = next(iter(model.planners.values()))
    scenarios: dict[str, Any] = {}
    if "embeddings" in model.planners:
        head = planner.heads[planner.info.layers[-1]]
        rows = embed_scenarios(vocab)
        served = list(planner.info.layers)
    else:
        head = planner.heads[planner.layout.default]
        rows = rerank_scenarios(vocab)
        served = [list(exit) for exit in planner.info.exits]
    for name, scenario in rows.items():
        items = [Item(ids, head.name, head.layer) for ids in scenario]
        many = len(items) > 1
        result = timed(
            model,
            items,
            args.warmup,
            max(args.iterations // 3, 5) if many else args.iterations,
        )
        if many:
            result["items_per_s"] = round(len(items) / result["mean_ms"] * 1000, 1)
        scenarios[name] = result
    record = {
        "package": args.package.name,
        "model": model.info.id,
        "model_sha256": model.info.model_sha256,
        "engine": args.engine,
        "served_exits": served,
        "device": args.device,
        "threads": args.threads,
        "torch": torch.__version__,
        "machine": platform.processor() or platform.machine(),
        "scenarios": scenarios,
    }
    receipt = getattr(model.engine_model, "receipt", None)
    if callable(receipt):
        record["receipt"] = receipt()
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record["scenarios"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
