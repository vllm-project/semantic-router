"""One-step reload parity for an M4b full checkpoint, through the inference path.

The checkpoint is loaded the way the 27B kernel readouts load a full
checkpoint (``training.model.infer``: ``DecisionModel.from_checkpoint``, FP32
parameters, ``.float().to(device).eval()``, inference mode, BF16 autocast, FP32
head, one row per call), after its ``checkpoint_fingerprint`` identity is
computed. On GPU the image's kernel runtime is required first, as in
``v2.27b.typed_collect_kernel``. The first ``--rows`` SELECT rows are scored and
compared with the trainer's in-process ``onestep.select32.jsonl``: PASS iff every
row is compared, no argmax changes and max |dp| <= ``--tolerance``.

Run from ``src/training/decision2`` as ``python3 -m v2.27b.m4b.reload_check``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from pathlib import Path

import torch

from training.model.data import load_partition
from training.model.decision_model import DecisionModel, encode
from training.model.infer import checkpoint_fingerprint

train_ff = importlib.import_module("v2.27b.m4b.train_ff")


def argmax(values: list[float]) -> int:
    return max(range(len(values)), key=values.__getitem__)


def compare(reference: list[dict], actual: list[dict]) -> dict:
    worst, changed, compared = 0.0, 0, 0
    for expected, got in zip(reference, actual):
        if expected["id"] != got["id"] or expected["keys"] != got["keys"]:
            raise ValueError(
                f"Row {got['id']} differs from the trainer's row order or keys"
            )
        worst = max(
            worst,
            max(
                abs(a - b)
                for a, b in zip(expected["probabilities"], got["probabilities"])
            ),
        )
        changed += argmax(expected["probabilities"]) != argmax(got["probabilities"])
        compared += 1
    return {
        "rows": compared,
        "argmax_changes": changed,
        "max_abs_probability_diff": worst,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", help="Name inside --run-dir (default: LATEST.json)"
    )
    parser.add_argument("--select", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=train_ff.SELECT32_ROWS)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    runtime = None
    if args.device == "cuda":
        runtime = importlib.import_module(
            "v2.27b.typed_collect_kernel"
        ).kernel_runtime()
    name = (
        args.checkpoint
        or json.loads((args.run_dir / "LATEST.json").read_text())["checkpoint"]
    )
    checkpoint = args.run_dir / name
    with (args.run_dir / "onestep.select32.jsonl").open(encoding="utf-8") as stream:
        reference = [json.loads(line) for line in stream if line.strip()][: args.rows]
    rows = load_partition(args.select, "select")[: args.rows]
    identity = checkpoint_fingerprint(checkpoint)
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    model, tokenizer = DecisionModel.from_checkpoint(checkpoint)
    model = model.float().to(device).eval()
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    items = [encode(row, tokenizer, args.max_length) for row in rows]
    actual = train_ff.select_probabilities(model, items, pad_id=pad_id, device=device)
    result = compare(reference, actual)
    passed = (
        result["rows"] == args.rows == len(rows)
        and result["argmax_changes"] == 0
        and math.isfinite(result["max_abs_probability_diff"])
        and result["max_abs_probability_diff"] <= args.tolerance
    )
    receipt = {
        "schema_version": "decision2-27b-m4b-reload/1",
        "checkpoint": name,
        "checkpoint_format": model.metadata.get("checkpoint_format"),
        "model_sha256": identity["model_sha256"],
        **result,
        "tolerance": args.tolerance,
        "passed": passed,
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu",
        "runtime": runtime,
    }
    fd = os.open(args.output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=1, sort_keys=True)
        stream.write("\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "runtime"}), flush=True)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
