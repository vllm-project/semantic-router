"""The legacy ONNX Runtime execution of the embedding and rerank graphs, as a performance baseline.

    python3 tools/embed_legacy_baseline.py --package DIR --task embed|rerank --graph onnx/model_fa.onnx \
        --provider ROCMExecutionProvider [--custom-ops LIB] [--device-id N] [--threads N] --output OUT.json

Runs where the legacy router ran (its image: the ONNX Runtime build and the
CK flash-attention operator library it shipped) and executes a package graph
the way the legacy ``onnx-binding`` did: one sequence per session run, the
pairs of a rerank request one after another, batch-one ``position_ids``.
Needs only NumPy and onnxruntime. The scenarios and synthetic token rows are
``tools/embed_bench.py``'s, so both sides see identical inputs.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
from embed_corpus import embed_scenarios, rerank_scenarios


def session(args: argparse.Namespace) -> Any:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if args.threads:
        options.intra_op_num_threads = args.threads
    if args.custom_ops:
        options.register_custom_ops_library(args.custom_ops)
    provider: Any = args.provider
    if args.provider != "CPUExecutionProvider":
        provider = (args.provider, {"device_id": str(args.device_id)})
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    return ort.InferenceSession(
        str(args.package / args.graph), options, providers=[provider]
    )


def request(model: Any, rows: list[tuple[int, ...]]) -> list[np.ndarray]:
    """One legacy call per row: batch one, no padding."""
    outputs = []
    for row in rows:
        ids = np.asarray([row], dtype=np.int64)
        feeds = {
            "input_ids": ids,
            "attention_mask": np.ones_like(ids),
            "position_ids": np.arange(ids.shape[1], dtype=np.int64)[None],
        }
        outputs.append(model.run(None, feeds)[0])
    return outputs


def timed(
    model: Any, rows: list[tuple[int, ...]], warmup: int, iterations: int
) -> dict[str, float]:
    for _ in range(warmup):
        request(model, rows)
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        request(model, rows)
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
    parser.add_argument("--task", choices=("embed", "rerank"), required=True)
    parser.add_argument("--graph", default="onnx/model.onnx")
    parser.add_argument("--provider", default="CPUExecutionProvider")
    parser.add_argument("--custom-ops", default=None)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import onnxruntime as ort

    model = session(args)
    vocab = int(json.loads((args.package / "config.json").read_text())["vocab_size"])
    rows = embed_scenarios(vocab) if args.task == "embed" else rerank_scenarios(vocab)
    scenarios: dict[str, Any] = {}
    for name, scenario in rows.items():
        many = len(scenario) > 1
        result = timed(
            model,
            scenario,
            args.warmup,
            max(args.iterations // 3, 5) if many else args.iterations,
        )
        if many:
            result["items_per_s"] = round(len(scenario) / result["mean_ms"] * 1000, 1)
        scenarios[name] = result
    record = {
        "package": args.package.name,
        "baseline": "legacy onnx-binding execution (one sequence per session run)",
        "graph": args.graph,
        "provider": model.get_providers()[0],
        "custom_ops": args.custom_ops,
        "onnxruntime": ort.__version__,
        "threads": args.threads,
        "machine": platform.processor() or platform.machine(),
        "scenarios": scenarios,
    }
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(scenarios))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
