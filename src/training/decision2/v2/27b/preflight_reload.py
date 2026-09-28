"""One-step save/reload parity for a LoRA Decision checkpoint.

Reloads ``checkpoint-N`` from disk with its pinned source, recomputes the first
``--rows`` SELECT examples exactly like the trainer's evaluator (FP32
parameters, BF16 autocast, FP32 head) and compares probabilities with the
trainer's own ``select-step-N-predictions.jsonl`` written before the save.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch

from training.model.decision_model import DecisionModel, collate, encode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--select", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tag = f"{args.step:07d}"
    checkpoint = args.run_dir / f"checkpoint-{tag}"
    saved = {}
    with (args.run_dir / f"select-step-{tag}-predictions.jsonl").open(
        encoding="utf-8"
    ) as stream:
        for line in stream:
            row = json.loads(line)
            saved[row["id"]] = row
    with args.select.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()][: args.rows]
    device = torch.device("cuda:0")
    model, tokenizer = DecisionModel.from_checkpoint(
        checkpoint, source_path=args.source_path
    )
    model = model.float().to(device).eval()
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    worst = 0.0
    changed = 0
    compared = 0
    with torch.inference_mode():
        for row in rows:
            item = encode(row, tokenizer, args.max_length)
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v)
                for k, v in collate([item], pad_id).items()
            }
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(**batch)
            probabilities = (
                logits.float().softmax(-1)[0, : len(item["keys"])].cpu().tolist()
            )
            reference = saved[row["id"]]["answer"]
            if reference["type"] == "noul":
                expected = {"true": reference["noul"], "false": 1 - reference["noul"]}
            else:
                expected = reference["probabilities"]
            actual = dict(zip(item["keys"], probabilities))
            worst = max(
                worst, max(abs(actual[key] - expected[key]) for key in item["keys"])
            )
            if max(actual, key=actual.get) != max(expected, key=expected.get):
                changed += 1
            compared += 1
    passed = (
        compared == args.rows
        and changed == 0
        and math.isfinite(worst)
        and worst <= args.tolerance
    )
    result = {
        "schema_version": "decision2-27b-reload-parity/1",
        "checkpoint": checkpoint.name,
        "rows": compared,
        "argmax_changes": changed,
        "max_abs_probability_diff": worst,
        "tolerance": args.tolerance,
        "passed": passed,
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "device_count": torch.cuda.device_count(),
    }
    fd = os.open(args.output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=1, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
