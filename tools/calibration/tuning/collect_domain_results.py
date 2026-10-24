#!/usr/bin/env python3
"""Score the domain classifier on held-out MMLU-Pro rows for calibration.

The rows are the quality baseline's domain split (``baseline_tasks.py``):
MMLU-Pro test questions whose source is not MMLU, since Vela Domain trains on
MMLU (#4300). Each row records the top label and its probability. Rows split
into calibration and held_out by a hash of the question id, and the script
writes both files plus a ``signal-calibration/v1`` manifest. The manifest binds
the model by the identity the model runtime computes for the same directory,
which its ``/v1/models`` card reports as ``model_sha256``. It never changes
router configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from vllm_sr_runtime.families.task_heads import package as task_heads
from vllm_sr_runtime.registry.artifacts import named_files

DATASET = "TIGER-Lab/MMLU-Pro"
DATASET_FILE = "data/test-00000-of-00001.parquet"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split(question_id: str) -> str:
    first = hashlib.sha256(question_id.encode("utf-8")).digest()[0]
    return "calibration" if first % 2 == 0 else "held_out"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    data_path = Path(
        hf_hub_download(
            DATASET, DATASET_FILE, repo_type="dataset", revision=args.dataset_revision
        )
    )
    # Batches of similar length pad less. Output order is fixed by id below.
    rows = sorted(
        (
            row
            for row in pq.read_table(data_path).to_pylist()
            if not str(row["src"]).startswith("ori_mmlu")
        ),
        key=lambda row: len(row["question"]),
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).eval()
    labels = [model.config.id2label[index] for index in range(model.config.num_labels)]

    results: dict[str, list[dict]] = {"calibration": [], "held_out": []}
    with torch.no_grad():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            print(f"{start}/{len(rows)}", file=sys.stderr, flush=True)
            encoded = tokenizer(
                [row["question"] for row in batch],
                truncation=True,
                max_length=512,
                padding=True,
                return_tensors="pt",
            )
            probabilities = torch.softmax(model(**encoded).logits.float(), dim=-1)
            scores, indices = probabilities.max(dim=-1)
            for row, score, index in zip(batch, scores, indices, strict=True):
                question_id = str(row["question_id"])
                results[_split(question_id)].append(
                    {
                        "id": question_id,
                        "category": row["category"],
                        "label": labels[int(index)],
                        "score": float(score),
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, items in results.items():
        items.sort(key=lambda item: int(item["id"]))
        (args.output_dir / f"{split}.json").write_text(
            json.dumps(items, indent=1) + "\n", encoding="utf-8"
        )
    manifest = {
        "schema_version": "signal-calibration/v1",
        "name": "vela-domain-label-correctness-v1",
        "family": "domain",
        "scale": "label_correctness/v1",
        "method": "isotonic",
        "dataset": {
            "name": DATASET,
            "version": args.dataset_revision,
            "digest": f"sha256:{_sha256(data_path)}",
        },
        "population": (
            "MMLU-Pro test questions whose src does not start with ori_mmlu, "
            "split into calibration and held_out by the first byte of "
            "sha256(question_id)"
        ),
        "outcome": "The top domain label equals the question's MMLU-Pro category",
        "model": {
            "id": args.model_id,
            "revision": args.model_revision,
            "labels": labels,
            "model_sha256": task_heads.identity(
                named_files(args.model_dir, task_heads.read(args.model_dir).files)
            ),
        },
        "operating_threshold": 0.5,
        "splits": {split: f"{split}.json" for split in results},
        "policy": {"rollback_identity": "uncalibrated-domain-probability"},
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print({split: len(items) for split, items in results.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
