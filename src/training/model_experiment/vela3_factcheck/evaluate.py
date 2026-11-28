"""Evaluate a Vela3 FactCheck adapter on the public suite and boundary probes."""

import argparse
import json
import math
import time
from collections import defaultdict
from functools import partial
from itertools import groupby
from pathlib import Path

import numpy as np
import torch
from vllm_srun.config import ModelConfig, ServeConfig
from vllm_srun.plugins.base import SurfacePlan
from vllm_srun.runtime import Runtime

from ...model_eval.provenance.emit import split_digest
from ...model_eval.provenance.metrics import roc_auc
from .data import NEUTRAL_THRESHOLD, POSITIVE, read_rows, suite_rows
from .package import ADAPTER, sha256
from .train import QUESTION, questions

SPLITS = (
    "dev",
    "test",
    "hold-no_robots",
    "hold-simpleqa",
    "hold-wildbench",
    "hold-shipped-factcheck",
    "fresh-halueval-wild",
    "fresh-mgsm",
    "fresh-mbpp",
    "fresh-truthfulqa",
    "fresh-mintaka",
)


def threshold_at_fpr(rows, cap=0.20):
    allowed = math.floor(cap * sum(not row["positive"] for row in rows))
    threshold = math.nextafter(max(row["score"] for row in rows), math.inf)
    for score, ties in groupby(
        sorted(rows, key=lambda row: -row["score"]), key=lambda row: row["score"]
    ):
        allowed -= sum(not row["positive"] for row in ties)
        if allowed < 0:
            break
        threshold = score
    return threshold


def rates(score, truth, threshold):
    selected = score >= threshold
    return {
        "tp": int((selected & truth).sum()),
        "positive": int(truth.sum()),
        "fp": int((selected & ~truth).sum()),
        "negative": int((~truth).sum()),
    }


def average_precision(score, truth):
    sorted_idx = np.argsort(-score, kind="stable")
    thresholds = np.flatnonzero(np.r_[np.diff(score[sorted_idx]) != 0, True])
    tp = np.cumsum(truth[sorted_idx])[thresholds]
    recall = tp / tp[-1]
    return float(np.sum(np.diff(np.r_[0, recall]) * tp / (thresholds + 1)))


def summarize(rows, threshold, *, published_threshold=None, bootstrap=False):
    score = np.array([row["score"] for row in rows])
    truth = np.array([row["positive"] for row in rows], dtype=bool)
    abstain = np.array([row["abstain"] for row in rows])
    both = truth.any() and not truth.all()
    bins = np.minimum((10 * score).astype(int), 9)
    report = {
        "auc": roc_auc(score.tolist(), truth.tolist()) if both else None,
        "average_precision": average_precision(score, truth) if both else None,
        "brier": float(np.mean((score - truth) ** 2)),
        "ece": float(
            sum(abs((score[bins == i] - truth[bins == i]).sum()) for i in range(10))
            / len(rows)
        ),
        "abstain_p50_p95": np.percentile(abstain, [50, 95]).tolist(),
        "neutral": rates(score, truth, NEUTRAL_THRESHOLD),
        "selected_on_dev": rates(score, truth, threshold),
    }
    if published_threshold is not None:
        report["published_operating_point"] = rates(score, truth, published_threshold)
    if bootstrap:
        groups = defaultdict(list)
        for index, row in enumerate(rows):
            groups[row["group"] or row["id"]].append(index)
        rng = np.random.default_rng(42)
        blocks = list(groups.values())
        sampled = defaultdict(list)
        for _ in range(1000):
            indices = np.concatenate(
                [blocks[i] for i in rng.integers(len(blocks), size=len(blocks))]
            )
            s, y = score[indices], truth[indices]
            for operating_point, cut in (
                ("neutral", NEUTRAL_THRESHOLD),
                ("selected_on_dev", threshold),
            ):
                if y.any():
                    sampled[f"recall_{operating_point}"].append(
                        float(np.mean(s[y] >= cut))
                    )
                if (~y).any():
                    sampled[f"fpr_{operating_point}"].append(
                        float(np.mean(s[~y] >= cut))
                    )
            if y.any() and not y.all():
                sampled["auc"].append(roc_auc(s.tolist(), y.tolist()))
        report["bootstrap_95"] = {
            metric: np.percentile(values, [2.5, 97.5]).tolist()
            for metric, values in sampled.items()
        }
    return report


def score_rows(model, placement, request, rows, batch_size):
    """Score using the runtime answer decoder and length-aware batching."""
    scored = []
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset : offset + batch_size]
        plans = [model.plan(row["text"], request) for row in batch]
        items = [item for plan in plans for item in plan.items]
        if len(items) != len(batch) or any(len(plan.items) != 1 for plan in plans):
            raise ValueError("expected one runtime input per FactCheck question")
        outputs = placement.accelerator.execute(
            placement.device, partial(model.networks.run, items, merged=True)
        )
        for row, plan, output in zip(batch, plans, outputs, strict=True):
            answer = model.finish_surface(
                SurfacePlan("decisions", plan.items, plan.input_tokens, plan), [output]
            )["answers"][QUESTION]
            scored.append(
                {
                    "id": row["id"],
                    "group": row.get("group"),
                    "lang": row.get("lang"),
                    "slice": row.get("slice"),
                    "positive": row["label"] == POSITIVE,
                    "score": answer["probabilities"][POSITIVE],
                    "abstain": answer["abstain_probability"],
                }
            )
    return scored


def pair_results(rows, threshold):
    groups = defaultdict(dict)
    for row in rows:
        if row["slice"] == "evidence":
            groups[row["group"]][row["positive"]] = row["score"]
    pairs = [(pair[False], pair[True]) for pair in groups.values()]
    return {
        "pairs": len(pairs),
        "ordered": sum(pos > neg for neg, pos in pairs),
        "both_correct_neutral": sum(
            neg < NEUTRAL_THRESHOLD <= pos for neg, pos in pairs
        ),
        "both_correct_dev": sum(neg < threshold <= pos for neg, pos in pairs),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--probes", type=Path, action="append", default=[])
    parser.add_argument("--split", choices=SPLITS[1:], action="append")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scores-out", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--shared", action="store_true")
    parser.add_argument("--published-threshold", type=float, default=0.9999)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    if not 0 <= args.published_threshold <= 1:
        parser.error("published-threshold must be in [0, 1]")
    runtime = Runtime(
        ServeConfig(
            models=(ModelConfig(model=str(args.model.resolve()), device=args.device),)
        )
    )
    started = time.perf_counter()
    try:
        runtime.load()
        served = runtime.lookup(None)
        model, placement = served.model, served.placement
        request = questions(model)
        if args.shared:
            model.router.roster.router_tasks.pop("factcheck")
        splits = ("dev", *dict.fromkeys(args.split)) if args.split else SPLITS
        inputs = {split: suite_rows(args.suite, split) for split in splits}
        inputs.update({path.stem: read_rows(path) for path in args.probes})
        raw = {}
        for name, rows in inputs.items():
            if not rows:
                raise ValueError(f"empty evaluation split: {name}")
            raw[name] = score_rows(model, placement, request, rows, args.batch_size)
            print(f"{name}: {len(rows)}", flush=True)
        peak_memory = (
            torch.cuda.max_memory_allocated(model.engine_model.device) / 1024**2
            if model.engine_model.device.type == "cuda"
            else None
        )
    finally:
        runtime.stop()
    threshold = threshold_at_fpr(raw["dev"])
    language = defaultdict(list)
    for row in raw.get("fresh-mgsm", ()):
        language[row["lang"]].append(row)
    probes = {}
    for path in args.probes:
        rows = raw[path.stem]
        probes[path.stem] = {
            **pair_results(rows, threshold),
            **{
                name: summarize([r for r in rows if r["slice"] == name], threshold)
                for name in sorted({r["slice"] for r in rows})
            },
        }
    report = {
        "adapter_sha256": sha256(
            args.model / ("router_adapter.safetensors" if args.shared else ADAPTER)
        ),
        "dev_threshold": threshold,
        "published_threshold": args.published_threshold,
        "batch_size": args.batch_size,
        "peak_cuda_memory_mib": peak_memory,
        "max_input_tokens": int(model.package.max_input_tokens),
        "model_sha256": model.info.model_sha256,
        "input_splits": {
            name: split_digest(
                (row["text"], int(row["label"] == POSITIVE)) for row in rows
            )
            for name, rows in inputs.items()
        },
        "wall_seconds": time.perf_counter() - started,
        "splits": {
            key: summarize(
                raw[key],
                threshold,
                published_threshold=args.published_threshold,
                bootstrap=key in {"test", "fresh-mgsm"},
            )
            for key in splits
        },
        "mgsm_by_language": {
            key: summarize(value, threshold) for key, value in sorted(language.items())
        },
        "probes": probes,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.scores_out is not None:
        with args.scores_out.open("x", encoding="utf-8") as file:
            json.dump(raw, file, allow_nan=False)
            print(file=file)
        report["scores_sha256"] = sha256(args.scores_out)
    with args.out.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2, allow_nan=False)
        print(file=file)


if __name__ == "__main__":
    main()
