"""Throughput of cross-request batching with and without batch-shape buckets, on traffic that never repeats.

    python3 tools/bucket_bench.py --package DIR --prompts NAME:PROMPTS.jsonl:COUNT ... --mode MODE
        --output OUT.json [--answers OUT.jsonl] [--concurrency 16] [--warmup 512] [--rows-bucket 2]
        [--length-bucket 64] [--graphable-only] [--device rocm:0] [--base-path DIR]

MODE is ``exact`` (one request per forward, the reference answers), ``batching`` (the profile as it is)
or ``buckets``: the batching profile's batches, each padded up to a bucketed shape before its forward,
rows to the next power of --rows-bucket with copies of the batch's last row and the padded length to a
multiple of --length-bucket with padding columns (with --graphable-only, only when the bucketed shape
stays within the graph-size cap). Padding rows and columns never change which rows are
answered, so the shapes a forward sees, and therefore its graphs, repeat across batches; the padding
compute is the price. The first --warmup prompts are answered once untimed, the rest once, timed, in
waves of --concurrency requests submitted together through the scheduler (2 ms window). No request
repeats, so a graph is replayed only when a later batch has the same shape. Reported: requests/s,
mean wave latency, the timed pass's graph captures / replays / eager forwards, and the padded tokens
per answered token.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gpu_parity import load
from vllm_srun.plugins.base import ForwardBatch, ForwardOutput
from vllm_srun.profiles.batching import BatchingProfile
from vllm_srun.profiles.exact import ExactProfile
from vllm_srun.scheduler.scheduler import Scheduler, SchedulerLimits


def bucket_rows(rows: int, base: int) -> int:
    size = 1
    while size < rows:
        size *= base
    return size


def install_buckets(
    model, rows_base: int, length_step: int, graphable_only: bool, tally: dict
) -> None:
    """Pad forwards up to a bucketed shape: extra rows copy the last row and are dropped afterwards."""
    import torch

    engine = model.engine_model
    forward = engine.forward
    pad_id = model.tokenizer.pad_id

    def bucketed(batch: ForwardBatch) -> ForwardOutput:
        count, length = batch.input_ids.shape
        rows = bucket_rows(count, rows_base)
        width = -(-length // length_step) * length_step
        if graphable_only and rows * width > engine.graphs.max_tokens:
            rows, width = count, length
        tally["answered"] += sum(batch.lengths)
        tally["padded"] += rows * width
        if batch.shared_prefix or (rows, width) == (count, length):
            return forward(batch)

        def grow(tensor: torch.Tensor, fill: int) -> torch.Tensor:
            out = tensor.new_full((rows, width), fill)
            out[:count, :length] = tensor
            out[count:, :length] = tensor[-1:]
            return out

        extra = rows - count
        output = forward(
            replace(
                batch,
                input_ids=grow(batch.input_ids, pad_id),
                attention_mask=grow(batch.attention_mask, 0),
                gather=torch.cat([batch.gather, batch.gather[-1:].expand(extra, -1)]),
                query=torch.cat([batch.query, batch.query[-1:].expand(extra)]),
                lengths=list(batch.lengths) + [batch.lengths[-1]] * extra,
            )
        )
        return ForwardOutput(
            gathered=output.gathered[:count], query=output.query[:count]
        )

    engine.forward = bucketed


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--package", required=True)
    parser.add_argument("--prompts", action="append", required=True)
    parser.add_argument(
        "--mode", choices=("exact", "batching", "buckets"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--answers", type=Path)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=512)
    parser.add_argument("--rows-bucket", type=int, default=2)
    parser.add_argument("--length-bucket", type=int, default=64)
    parser.add_argument("--graphable-only", action="store_true")
    parser.add_argument("--device", default="rocm:0")
    parser.add_argument("--base-path")
    args = parser.parse_args()
    args.no_graphs = args.no_fused = False
    import torch

    prompts = []
    for spec in args.prompts:
        name, path, count = spec.split(":")
        with open(path, encoding="utf-8") as source:
            prompts.extend(
                {"panel": name, **json.loads(line)}
                for line in list(source)[: int(count)]
            )
    model = load(args)
    tally = {"answered": 0, "padded": 0}
    if args.mode == "buckets":
        install_buckets(
            model, args.rows_bucket, args.length_bucket, args.graphable_only, tally
        )
    profile = ExactProfile() if args.mode == "exact" else BatchingProfile()
    scheduler = Scheduler(
        model,
        {profile.name: profile},
        SchedulerLimits(max_queue=4096, batch_window_ms=2.0),
    )
    scheduler.start()
    plans = [model.plan(prompt["state"], prompt["questions"]) for prompt in prompts]
    graphs = model.engine_model.graphs

    def waves(indices: list[int]) -> tuple[list[float], dict]:
        latencies, answers = [], {}
        for start in range(0, len(indices), args.concurrency):
            wave = indices[start : start + args.concurrency]
            tick = time.perf_counter()
            futures = [
                scheduler.submit(plans[i].items, deadline=None, profile=profile.name)
                for i in wave
            ]
            for i, future in zip(wave, futures, strict=True):
                answers[i] = future.result()
            latencies.append(1000 * (time.perf_counter() - tick))
        return latencies, answers

    waves(list(range(min(args.warmup, len(prompts)))))
    tally.update(answered=0, padded=0)
    timed = list(range(min(args.warmup, len(prompts)), len(prompts)))
    before = dict(graphs.receipt())
    torch.cuda.synchronize()
    started = time.perf_counter()
    latencies, results = waves(timed)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    after = graphs.receipt()
    scheduler.stop()
    if args.answers is not None:
        with args.answers.open("x", encoding="utf-8") as sink:
            for i in timed:
                answers = {
                    item.question_id: model.answer(item, results[i][index])
                    for index, item in enumerate(plans[i].items)
                }
                row = {
                    "panel": prompts[i]["panel"],
                    "id": prompts[i]["id"],
                    "answers": answers,
                }
                sink.write(json.dumps(row) + "\n")
    result = {
        "schema": "model-runtime-bucket-bench/1",
        "model": model.info.id,
        "device": model.engine_model.device_info.name,
        "mode": args.mode,
        "concurrency": args.concurrency,
        "buckets": (
            {
                "rows": f"powers of {args.rows_bucket}",
                "length": args.length_bucket,
                "graphable_only": args.graphable_only,
            }
            if args.mode == "buckets"
            else None
        ),
        "requests": len(timed),
        "seconds": seconds,
        "requests_per_s": len(timed) / seconds,
        "wave_ms_mean": statistics.mean(latencies),
        "graphs": {
            key: after[key] - before.get(key, 0)
            for key in ("captures", "replays", "eager", "full")
        }
        | {"cached": after["cached"]},
        "padded_per_answered_token": (
            tally["padded"] / tally["answered"] if tally["answered"] else None
        ),
    }
    with args.output.open("x", encoding="utf-8") as sink:
        json.dump(result, sink, indent=1)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
