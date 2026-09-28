"""Latency, throughput and candidate-cache cost of joint versus disaggregated readouts.

Timing only: heads are randomly initialized and outputs are discarded. Paths
per request:

* ``joint``: one causal pass over the native prompt with every option;
* ``disaggregated_cold``: the state text plus every candidate text encoded;
* ``disaggregated_warm``: candidate vectors cached, only the state encoded;
* ``cache_hit``: state and candidates cached, head arithmetic only.

Option-count scaling uses real SELECT states with synthetic option lists.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from . import pins
from .extract import LayerTap, forward_layers, load_backbone
from .render import candidate_texts, state_text

OPTION_COUNTS = (2, 4, 10, 32, 128, 255)
TOPICS = (
    "billing disputes",
    "password resets",
    "shipping delays",
    "refund status",
    "account closure",
    "fraud reports",
    "plan upgrades",
    "data export",
    "API quotas",
    "invoice copies",
    "tax forms",
    "device setup",
    "outage reports",
    "privacy requests",
    "contract terms",
)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))]


def synthetic_row(row: dict[str, Any], count: int) -> dict[str, Any]:
    options = [
        {
            "key": f"route_{i:03d}",
            "description": f"Team {i}: handles {TOPICS[i % len(TOPICS)]} (queue {i // len(TOPICS)})",
        }
        for i in range(count)
    ]
    return {**row, "task_type": "choice", "options": options, "label": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, choices=sorted(pins.SOURCES))
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--select", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requests-per-type", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()

    import torch

    from training.model.data import load_partition
    from training.model.decision_model import CandidateHead, encode

    from .heads import DualProjection

    pins.verify_source(args.source, args.source_path)
    pins.verify_data("select", args.select)
    try:
        import fla  # noqa: F401

        gated_delta = "fla"
    except ImportError:
        gated_delta = "torch-reference"
    device = torch.device("cuda:0")
    backbone, tokenizer, parameters = load_backbone(args.source, args.source_path)
    backbone = backbone.to(device)
    pad = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    tap = LayerTap(backbone, (32,))
    torch.manual_seed(0)
    joint_head = CandidateHead(4096, 256).to(device).eval()
    dual = DualProjection(4096).to(device).eval()
    rows = load_partition(args.select, "select")
    sample = []
    for kind in ("choice", "noul", "score"):
        sample += [r for r in rows if r["task_type"] == kind][: args.requests_per_type]

    def sync() -> None:
        torch.cuda.synchronize(device)

    def encode_texts(ids_list: list[list[int]]):
        hidden, _ = forward_layers(backbone, tap, ids_list, pad, device, (32,))
        last = torch.tensor([len(ids) - 1 for ids in ids_list], device=device)
        return hidden[32][torch.arange(len(ids_list), device=device), last]

    def joint(row):
        encoded = encode(row, tokenizer, 1 << 20)
        hidden, _ = forward_layers(backbone, tap, [encoded["ids"]], pad, device, (32,))
        positions = torch.tensor(
            [*encoded["candidate_positions"], encoded["query_position"]], device=device
        )
        vectors = hidden[32][0].index_select(0, positions)
        joint_head(vectors[None, :-1], vectors[None, -1]).softmax(-1)
        return len(encoded["ids"])

    def disaggregated(row, cached_candidates=None, cached_state=None):
        tokens = 0
        if cached_state is None:
            ids = tokenizer.encode(state_text(row), add_special_tokens=False)
            state = dual.encode_state(encode_texts([ids]))
            tokens += len(ids)
        else:
            state = cached_state
        if cached_candidates is None:
            texts = [
                tokenizer.encode(text, add_special_tokens=False)
                for text in candidate_texts(row)
            ]
            candidates = dual.encode_action(encode_texts(texts))
            tokens += sum(len(t) for t in texts)
        else:
            candidates = cached_candidates
        (dual.scale() * candidates @ state[0]).softmax(-1)
        return tokens, state, candidates

    def timed(fn, *fn_args) -> tuple[float, Any]:
        sync()
        start = time.perf_counter()
        with torch.inference_mode():
            result = fn(*fn_args)
        sync()
        return (time.perf_counter() - start) * 1000, result

    def measure(requests: list[dict[str, Any]]) -> dict[str, Any]:
        stats: dict[str, dict[str, list[float]]] = {
            path: {"ms": [], "tokens": []}
            for path in (
                "joint",
                "disaggregated_cold",
                "disaggregated_warm",
                "cache_hit",
            )
        }
        for row in requests:
            for _ in range(2):
                timed(joint, row)
                timed(disaggregated, row)
            for _ in range(args.repeats):
                ms, tokens = timed(joint, row)
                stats["joint"]["ms"].append(ms)
                stats["joint"]["tokens"].append(tokens)
                ms, (tokens, state, candidates) = timed(disaggregated, row)
                stats["disaggregated_cold"]["ms"].append(ms)
                stats["disaggregated_cold"]["tokens"].append(tokens)
                ms, (tokens, _, _) = timed(disaggregated, row, candidates)
                stats["disaggregated_warm"]["ms"].append(ms)
                stats["disaggregated_warm"]["tokens"].append(tokens)
                ms, (tokens, _, _) = timed(disaggregated, row, candidates, state)
                stats["cache_hit"]["ms"].append(ms)
                stats["cache_hit"]["tokens"].append(tokens)
        return {
            path: {
                "p50_ms": percentile(values["ms"], 0.5),
                "p95_ms": percentile(values["ms"], 0.95),
                "mean_tokens": statistics.mean(values["tokens"]),
                "n": len(values["ms"]),
            }
            for path, values in stats.items()
        }

    started = time.perf_counter()
    report: dict[str, Any] = {
        "source": args.source,
        "text_parameters": parameters,
        "gated_delta_implementation": gated_delta,
        "precision": "FP32 parameters, BF16 autocast",
        "runtime": {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "device": torch.cuda.get_device_name(0),
        },
        "code_commit": args.code_commit,
        "select_requests": {},
        "option_scaling": {},
        "throughput": {},
    }
    for kind in ("choice", "noul", "score"):
        report["select_requests"][kind] = measure(
            [r for r in sample if r["task_type"] == kind]
        )
    anchors = [r for r in sample if r["task_type"] == "choice"][:3]
    for count in OPTION_COUNTS:
        torch.cuda.reset_peak_memory_stats(device)
        report["option_scaling"][str(count)] = measure(
            [synthetic_row(r, count) for r in anchors]
        )
        report["option_scaling"][str(count)]["peak_allocated_gib"] = (
            torch.cuda.max_memory_allocated(device) / 2**30
        )
    for batch in (1, 8, 32):
        chunk = sample[:batch] if batch <= len(sample) else sample
        joint_ids = [encode(r, tokenizer, 1 << 20)["ids"] for r in chunk]
        state_ids = [
            tokenizer.encode(state_text(r), add_special_tokens=False) for r in chunk
        ]
        for name, ids in (
            ("joint_padded_batch", joint_ids),
            ("disaggregated_warm_state_batch", state_ids),
        ):
            timings = []
            for repeat in range(args.repeats + 1):
                ms, _ = timed(
                    lambda: forward_layers(backbone, tap, ids, pad, device, (32,))
                )
                if repeat:
                    timings.append(ms)
            median = statistics.median(timings)
            report["throughput"].setdefault(name, {})[str(batch)] = {
                "median_batch_ms": median,
                "requests_per_second": len(ids) * 1000 / median,
                "tokens_per_batch": sum(len(i) for i in ids),
            }
    report["wall_seconds"] = time.perf_counter() - started
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / "bench.json"
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "bench": str(path),
                "sha256": pins.file_sha256(path),
                "wall_seconds": report["wall_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
