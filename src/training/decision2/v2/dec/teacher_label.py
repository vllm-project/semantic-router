"""Label rights-clean TRAIN rows with a pinned own-family Decision 1.0 teacher.

The teacher is loaded through the shared ``from_decision1`` path and scores the
same shared rendering the student trains on. Probabilities use the teacher
package's published per-type temperatures. Output rows carry only the TRAIN
ID, its canonical input hash and the option-key distribution; the manifest
records teacher identity, agreement with TRAIN labels by type and validity.
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


def softmax(values: list[float], temperature: float) -> list[float]:
    scaled = [value / temperature for value in values]
    top = max(scaled)
    exps = [math.exp(value - top) for value in scaled]
    total = math.fsum(exps)
    return [value / total for value in exps]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-path", type=Path, required=True)
    parser.add_argument("--teacher-repo", required=True)
    parser.add_argument("--teacher-revision", required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = load_partition(args.train, "train")
    temperatures = teacher_temperatures(args.teacher_path)
    source = source_fingerprint(args.teacher_path)
    device = torch.device("cuda:0")
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
            "teacher_source_fingerprint": source,
            "teacher_temperatures": temperatures,
            "train_sha256": file_sha256(args.train),
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
