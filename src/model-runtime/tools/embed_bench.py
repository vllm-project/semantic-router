"""Engine-level latency and throughput of the embedding and rerank models (embed workstream).

    python3 tools/embed_bench.py --package DIR --engine native|onnxruntime --task embed|rerank \
        --output OUT.json [--device cpu|rocm:0] [--threads N] [--layout padded|packed]
        [--iterations N] [--warmup N] [--exit LAYER] [--dimension D]

Runs the package's backbone through one engine with synthetic token rows (no
tokenizer, so every engine sees identical inputs) and times the forward plus
the head readout per request:

* ``embed``: one text of 16, 64, 256 and 1,024 tokens (latency p50 / p95 per
  request), and a batch of 32 texts of 16-256 tokens (one request; texts/s);
* ``rerank``: one query of 12 tokens with 10 and 50 documents of 64-192 tokens
  (all pairs of a request in one forward; pairs/s).

The native engine runs the package's ModernBERT or Qwen3 backbone (``padded``
or ``packed`` layout); the onnxruntime engine runs the package's exit graph.
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

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vllm_sr_runtime.accel.cpu import CPUAccelerator  # noqa: E402
from vllm_sr_runtime.accel.rocm import ROCmAccelerator  # noqa: E402
from vllm_sr_runtime.engines.native.engine import NativeEngine  # noqa: E402
from vllm_sr_runtime.engines.native.models.modernbert import (  # noqa: E402
    packed_layout,
    padded_layout,
)
from vllm_sr_runtime.engines.onnxruntime.engine import OnnxRuntimeEngine  # noqa: E402
from vllm_sr_runtime.heads.pooled import PooledHead  # noqa: E402
from vllm_sr_runtime.heads.relevance import RelevanceHead  # noqa: E402
from vllm_sr_runtime.plugins.base import (  # noqa: E402
    BackboneSpec,
    DeviceInfo,
    DtypePolicy,
    EncoderBatch,
    EngineOptions,
    ModelSpec,
)

EMBED_LENGTHS = (16, 64, 256, 1024)
RERANK_DOCUMENTS = (10, 50)
BATCH = 32
SPECIAL_IDS = 8


def rows_of(lengths: list[int], vocab: int, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    return [
        [2, *rng.integers(SPECIAL_IDS, vocab, length - 2).tolist(), 1]
        for length in lengths
    ]


def padded_tensors(rows: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(row) for row in rows)
    ids = torch.zeros(len(rows), width, dtype=torch.long)
    mask = torch.zeros(len(rows), width, dtype=torch.long)
    for index, row in enumerate(rows):
        ids[index, : len(row)] = torch.tensor(row)
        mask[index, : len(row)] = 1
    return ids, mask


class Runner:
    """One engine, one package, one task: ``run(rows)`` is one request's forward and readout."""

    def __init__(self, args: argparse.Namespace):
        self.root = args.package
        self.config = json.loads((self.root / "config.json").read_text())
        self.task, self.layout = args.task, args.layout
        index = None if args.device == "cpu" else int(args.device.split(":")[1])
        accelerator = CPUAccelerator() if index is None else ROCmAccelerator()
        self.device = DeviceInfo(
            accelerator=args.device.split(":")[0], index=index, name=args.device
        )
        if self.task == "embed":
            self.head = PooledHead.detect(self.root, self.config)
            self.exit = args.exit or self.head.layers[-1]
            self.dimension = args.dimension or self.head.dimensions[0]
            graphs = (
                {"default": self.head.graphs[self.exit]}
                if self.exit in self.head.graphs
                else {}
            )
        else:
            self.head = RelevanceHead.detect(self.root, self.config)
            self.exit = (
                args.exit or self.head.default[0],
                args.dimension or self.head.default[1],
            )
            graphs = (
                {"default": self.head.graphs[self.exit]}
                if self.exit in self.head.graphs
                else {}
            )
        spec = ModelSpec(
            name=self.root.name,
            backbone=BackboneSpec(
                model_type=self.config["model_type"],
                config=self.config,
                weight_files=(self.root / "model.safetensors",),
            ),
            dtype=DtypePolicy(autocast=None, bf16_resident=False),
            max_input_tokens=int(self.config["max_position_embeddings"]),
            graphs=graphs,
            encoder=True,
        )
        engine = NativeEngine() if args.engine == "native" else OnnxRuntimeEngine()
        reason = engine.supports(spec, self.device)
        if reason:
            raise SystemExit(reason)
        self.model = engine.load(
            spec, accelerator, self.device, EngineOptions(threads=args.threads)
        )
        self.engine = args.engine
        if self.task == "rerank" and self.engine == "native":
            self.scorers = self.head.load([self.exit], self.model.device)

    def run(self, rows: list[list[int]]) -> Any:
        ids, mask = padded_tensors(rows)
        if self.engine == "onnxruntime":
            out = self.model.encode(EncoderBatch(input_ids=ids, attention_mask=mask))
            if self.task == "rerank":
                return out.outputs["logits"][:, 0]
            return self.head.readout(
                out.outputs["last_hidden_state"], mask, self.dimension
            )
        backbone, device = self.model.backbone, self.model.device
        layer = self.exit if self.task == "embed" else self.exit[0]
        with torch.inference_mode():
            if self.config["model_type"] != "modernbert":
                hidden = backbone(ids.to(device), mask.to(device))
                return self.head.readout(hidden, mask.to(device), self.dimension).cpu()
            if self.layout == "packed":
                lengths = [len(row) for row in rows]
                flat = torch.tensor(
                    [token for row in rows for token in row], device=device
                )
                layout = packed_layout(lengths, backbone.window, device)
                hidden = layout.to_rows(backbone.encode(flat, layout, (layer,))[layer])
            else:
                layout = padded_layout(mask, *ids.shape, backbone.window, device)
                hidden = backbone.encode(ids.to(device), layout, (layer,))[layer]
            if self.task == "rerank":
                return self.head.readout(self.scorers, hidden[:, 0], self.exit).cpu()
            return self.head.readout(hidden, mask.to(device), self.dimension).cpu()


def timed(
    runner: Runner, rows: list[list[int]], warmup: int, iterations: int
) -> dict[str, float]:
    for _ in range(warmup):
        runner.run(rows)
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        runner.run(rows)
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
    parser.add_argument("--task", choices=("embed", "rerank"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--layout", choices=("padded", "packed"), default="padded")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--exit", type=int, default=None)
    parser.add_argument("--dimension", type=int, default=None)
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    runner = Runner(args)
    vocab = int(runner.config["vocab_size"])
    scenarios: dict[str, Any] = {}
    if args.task == "embed":
        for length in EMBED_LENGTHS:
            scenarios[f"single/{length}"] = timed(
                runner, rows_of([length], vocab, length), args.warmup, args.iterations
            )
        lengths = np.random.default_rng(7).integers(16, 257, BATCH).tolist()
        batch = timed(
            runner,
            rows_of(lengths, vocab, 7),
            args.warmup,
            max(args.iterations // 3, 5),
        )
        batch["items_per_s"] = round(BATCH / batch["mean_ms"] * 1000, 1)
        scenarios[f"batch/{BATCH}x16-256"] = batch
    else:
        query = rows_of([12], vocab, 1)[0][:-1]
        for count in RERANK_DOCUMENTS:
            lengths = np.random.default_rng(count).integers(64, 193, count).tolist()
            pairs = [
                query + document[1:] for document in rows_of(lengths, vocab, count)
            ]
            result = timed(runner, pairs, args.warmup, max(args.iterations // 3, 5))
            result["pairs_per_s"] = round(count / result["mean_ms"] * 1000, 1)
            scenarios[f"query+{count}docs"] = result
    record = {
        "package": args.package.name,
        "engine": args.engine,
        "layout": args.layout if args.engine == "native" else "graph",
        "task": args.task,
        "exit": runner.exit,
        "device": args.device,
        "threads": args.threads,
        "torch": torch.__version__,
        "machine": platform.processor() or platform.machine(),
        "scenarios": scenarios,
    }
    if args.engine == "onnxruntime":
        record["receipt"] = runner.model.receipt()
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record["scenarios"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
