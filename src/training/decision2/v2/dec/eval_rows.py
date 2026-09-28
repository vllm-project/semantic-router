"""Score a labeled held-out slice (for example an arm's AHO rows) with one checkpoint.

Rows use the SELECT partition contract. Uses the trainer's own ``evaluate``
(raw probabilities, family metrics) and writes predictions plus metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from training.model.data import canonical, file_sha256, load_partition
from training.model.decision_model import DecisionModel, encode
from training.model.source import source_fingerprint
from training.model.train import evaluate

from .dec_model import dec_fingerprint, load_dec_checkpoint
from .runtime_check import require_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--checkpoint", type=Path)
    target.add_argument(
        "--package", type=Path, help="Untouched Decision 1.0 package (1.0 control)"
    )
    parser.add_argument("--source-path", type=Path)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    runtime = require_runtime()
    rows = load_partition(args.rows, "select")
    if args.package is not None:
        model, tokenizer = DecisionModel.from_decision1(args.package, 256)
        identity = source_fingerprint(args.package)["files_sha256"]
        model_sha = hashlib.sha256(canonical(identity).encode("utf-8")).hexdigest()
    else:
        model, tokenizer = load_dec_checkpoint(args.checkpoint, args.source_path)
        model_sha = dec_fingerprint(args.checkpoint, args.source_path)["model_sha256"]
    model = model.float().to(torch.device("cuda:0"))
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    items = [encode(row, tokenizer, args.max_length) for row in rows]
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = evaluate(
        model,
        items,
        pad_id=pad_id,
        batch_size=args.batch_size,
        device=torch.device("cuda:0"),
        output=args.output,
        tag=args.tag,
    )
    receipt = {
        "rows_sha256": file_sha256(args.rows),
        "model_sha256": model_sha,
        "tag": args.tag,
        "correct": metrics["correct"],
        "n": metrics["n"],
        "family_macro_accuracy": metrics["family_macro_accuracy"],
        "runtime": runtime,
    }
    (args.output / f"{args.tag}-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
