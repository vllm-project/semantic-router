"""Label rights-clean TRAIN rows with a pinned own-family teacher.

A Decision 1.0 teacher is loaded through the shared ``from_decision1`` path
and uses its package's published per-type temperatures; a decoder-track full
checkpoint (``--teacher-kind dec``, for example a released 2.0 soup) uses the
per-type temperatures of a calibration report bound to that checkpoint's
inference identity. Either scores the same shared rendering the student trains
on. Output rows carry only the TRAIN ID, its canonical input hash and the
option-key distribution; the manifest records teacher identity, agreement with
TRAIN labels by type and validity.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from training.model.data import file_sha256, load_partition
from training.model.decision_model import DecisionModel, collate, encode
from training.model.source import source_fingerprint
from training.model.train import atomic_json

from .runtime_check import require_runtime

LABEL_VERSION = "dec-own-teacher-labels/1"
TOKEN_BUDGET = 24_000


def teacher_temperatures(model_path: Path) -> dict[str, float]:
    report = json.loads((model_path / "temperature.json").read_text(encoding="utf-8"))
    temperatures = report.get("temperatures")
    if not isinstance(temperatures, dict) or set(temperatures) != {
        "choice",
        "noul",
        "score",
    }:
        raise ValueError("Teacher package lacks per-type temperatures")
    if any(
        type(v) not in (int, float) or not math.isfinite(v) or v <= 0
        for v in temperatures.values()
    ):
        raise ValueError("Teacher temperatures must be positive and finite")
    return {key: float(value) for key, value in temperatures.items()}


def calibration_temperatures(path: Path, model_sha256: str) -> dict[str, float]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("model_sha256") != model_sha256:
        raise ValueError("Calibration report belongs to a different checkpoint")
    temperatures = report.get("temperature_by_type")
    if not isinstance(temperatures, dict) or set(temperatures) != {
        "choice",
        "noul",
        "score",
    }:
        raise ValueError("Calibration report lacks per-type temperatures")
    if any(
        type(v) not in (int, float) or not math.isfinite(v) or v <= 0
        for v in temperatures.values()
    ):
        raise ValueError("Calibration temperatures must be positive and finite")
    return {key: float(value) for key, value in temperatures.items()}


def softmax(values: list[float], temperature: float) -> list[float]:
    scaled = [value / temperature for value in values]
    top = max(scaled)
    exps = [math.exp(value - top) for value in scaled]
    total = math.fsum(exps)
    return [value / total for value in exps]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-path", type=Path, required=True)
    parser.add_argument(
        "--teacher-kind", choices=("decision1", "dec"), default="decision1"
    )
    parser.add_argument(
        "--teacher-calibration",
        type=Path,
        help="dec teachers: calibration report (temperature_by_type) of that checkpoint",
    )
    parser.add_argument("--teacher-repo", required=True)
    parser.add_argument("--teacher-revision", required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Label only TRAIN rows whose position modulo --shard-count equals this",
    )
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("need 0 <= --shard-index < --shard-count")
    if (args.teacher_kind == "dec") != bool(args.teacher_calibration):
        parser.error("--teacher-calibration goes with --teacher-kind dec (only)")
    if args.output.exists():
        raise FileExistsError(args.output)
    runtime = require_runtime()
    rows = [
        row
        for position, row in enumerate(load_partition(args.train, "train"))
        if position % args.shard_count == args.shard_index
    ]
    device = torch.device("cuda:0")
    if args.teacher_kind == "dec":
        from .dec_model import dec_fingerprint, load_dec_checkpoint

        source = dec_fingerprint(args.teacher_path, None)["model_sha256"]
        temperatures = calibration_temperatures(args.teacher_calibration, source)
        model, tokenizer = load_dec_checkpoint(args.teacher_path, None)
    else:
        temperatures = teacher_temperatures(args.teacher_path)
        source = source_fingerprint(args.teacher_path)
        model, tokenizer = DecisionModel.from_decision1(args.teacher_path, 256)
    model = model.float().to(device).eval()
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    items = [encode(row, tokenizer, args.max_length) for row in rows]
    order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))
    batches: list[list[int]] = []
    current: list[int] = []
    for index in order:
        width = max(
            [len(items[i]["ids"]) for i in current] + [len(items[index]["ids"])]
        )
        if current and width * (len(current) + 1) > TOKEN_BUDGET:
            batches.append(current)
            current = []
        current.append(index)
    if current:
        batches.append(current)
    distributions: dict[int, list[float]] = {}
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_indices in batches:
            batch_items = [items[i] for i in batch_indices]
            batch = {
                key: (
                    value.to(device, non_blocking=True)
                    if torch.is_tensor(value)
                    else value
                )
                for key, value in collate(batch_items, pad_id).items()
            }
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(**batch)
            for index, item, values in zip(
                batch_indices, batch_items, logits.float().cpu().tolist()
            ):
                raw = values[: len(item["keys"])]
                if not all(math.isfinite(v) for v in raw):
                    raise RuntimeError(f"{item['id']}: nonfinite teacher logit")
                distributions[index] = softmax(raw, temperatures[item["task_type"]])
    seconds = time.perf_counter() - started
    agreement: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    confidence: dict[str, float] = defaultdict(float)
    pending = args.output.with_name(args.output.name + ".pending")
    with pending.open("x", encoding="utf-8") as stream:
        for index, row in enumerate(rows):
            probs = distributions[index]
            keys = items[index]["keys"]
            best = max(range(len(probs)), key=probs.__getitem__)
            stats = agreement[row["task_type"]]
            stats[0] += int(best == row["label"])
            stats[1] += 1
            confidence[row["task_type"]] += max(probs)
            stream.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "input_sha256": row["input_sha256"],
                        "teacher_probs": dict(zip(keys, probs)),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, args.output)
    atomic_json(
        args.output.with_name(args.output.name + ".manifest.json"),
        {
            "label_version": LABEL_VERSION,
            "teacher_repo": args.teacher_repo,
            "teacher_revision": args.teacher_revision,
            "teacher_kind": args.teacher_kind,
            "teacher_source_fingerprint": source,
            "teacher_calibration_sha256": (
                file_sha256(args.teacher_calibration)
                if args.teacher_calibration
                else None
            ),
            "teacher_temperatures": temperatures,
            "train_sha256": file_sha256(args.train),
            "shard": {"index": args.shard_index, "count": args.shard_count},
            "rows": len(rows),
            "output_sha256": file_sha256(args.output),
            "train_label_agreement": {
                kind: {"correct": c, "n": n, "accuracy": c / n}
                for kind, (c, n) in sorted(agreement.items())
            },
            "mean_max_probability": {
                kind: confidence[kind] / agreement[kind][1]
                for kind in sorted(agreement)
            },
            "max_length": args.max_length,
            "seconds": seconds,
            "torch_version": torch.__version__,
            "device_name": torch.cuda.get_device_name(device),
            "runtime": runtime,
        },
    )
    print(
        json.dumps(
            {"rows": len(rows), "seconds": seconds, "agreement": dict(agreement)}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
