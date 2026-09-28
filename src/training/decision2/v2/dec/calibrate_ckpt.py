"""Fit per-type CAL temperatures for one frozen checkpoint (for example a soup).

``calibrate_dec`` calibrates the SELECT-chosen BEST of a completed run on the
CAL partition that run audited. A release artifact that is not a single run's
BEST (a seed soup) or that must use a newer clean CAL (CAL698) is calibrated
here: same shared fitting code and report fields, bound to the checkpoint's
inference identity, so ``infer_dec`` accepts it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from training.model.calibration import CALIBRATION_VERSION, fit_report
from training.model.data import canonical, file_sha256, load_partition


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-path", type=Path)
    parser.add_argument("--cal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cal_rows = load_partition(args.cal, "cal")

    import torch

    from training.model.decision_model import collate, encode

    from .dec_model import dec_fingerprint, load_dec_checkpoint
    from .runtime_check import require_runtime

    runtime = require_runtime()
    identity = dec_fingerprint(args.checkpoint, args.source_path)
    device = torch.device("cuda:0")
    model, tokenizer = load_dec_checkpoint(args.checkpoint, args.source_path)
    model = model.float().to(device).eval()
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    encoded = [encode(row, tokenizer, args.max_length) for row in cal_rows]
    records = []
    with torch.inference_mode():
        for start in range(0, len(encoded), args.batch_size):
            items = encoded[start : start + args.batch_size]
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in collate(items, pad_id).items()
            }
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(**batch)
            for item, values in zip(items, logits):
                records.append(
                    {
                        "id": item["id"],
                        "task_type": item["task_type"],
                        "label": item["label"],
                        "logits": values[: len(item["keys"])].float().cpu().tolist(),
                    }
                )
    report = {
        "calibration_version": CALIBRATION_VERSION,
        "fit_split": "cal",
        "selection_policy": "frozen_checkpoint",
        "model_sha256": identity["model_sha256"],
        "checkpoint_sha256": _digest(
            {
                name.removeprefix("checkpoint/"): sha
                for name, sha in identity["files_sha256"].items()
                if not name.startswith("source/")
            }
        ),
        "cal_sha256": file_sha256(args.cal),
        "logits_sha256": _digest(records),
        "code_sha256": {
            name: file_sha256(Path(__file__).with_name(name))
            for name in ("calibrate_ckpt.py", "dec_model.py")
        },
        "inference": {
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "device": "cuda:0",
            "runtime": runtime,
        },
        **fit_report(records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pending = args.output.with_name(args.output.name + ".pending")
    with pending.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "temperature_by_type": report["temperature_by_type"],
            }
        )
    )


if __name__ == "__main__":
    main()
