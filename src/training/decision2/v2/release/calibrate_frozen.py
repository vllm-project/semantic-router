"""Fit per-type CAL temperatures for a frozen full Qwen checkpoint (for example a 0.6B soup).

``v2.dec.calibrate_ckpt`` does this for decoder checkpoints and requires the
Qwen3.5 kernels. This is the same shared fitting code and report contract for a
plain ``training.model`` full checkpoint (``checkpoint_fingerprint``,
``DecisionModel.from_checkpoint``) with the scored execution: BF16 backbone,
FP32 head, no truncation, one CAL question per forward pass by default (the
package runtime batches the questions of one state, and every CAL row is one
question). The report is accepted by ``training.model.infer --calibration``
and by the package runtime.

An adapter (``peft-lora/1``) checkpoint needs ``--source-path`` for its pinned
base. ``--require-kernels`` fails before loading unless
``v2.dec.runtime_check.require_runtime`` passes and records that identity as
``inference.kernel_runtime``.

    python3 -m v2.release.calibrate_frozen --checkpoint CKPT --cal CAL698.jsonl \
        --cal-sha256 H --output calibration.json [--logits logits.jsonl] \
        [--source-path BASE] [--require-kernels]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from training.model.calibration import CALIBRATION_VERSION, fit_report
from training.model.data import canonical, file_sha256


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def build_report(
    records: list[dict[str, Any]],
    identity: dict[str, Any],
    cal_sha256: str,
    inference: dict[str, Any],
) -> dict[str, Any]:
    """Frozen-checkpoint calibration report from CAL logits (pure; no torch)."""
    return {
        "calibration_version": CALIBRATION_VERSION,
        "fit_split": "cal",
        "selection_policy": "frozen_checkpoint",
        "model_sha256": identity["model_sha256"],
        "checkpoint_sha256": _digest(identity["files_sha256"]),
        "cal_sha256": cal_sha256,
        "logits_sha256": _digest(records),
        "code_sha256": {"calibrate_frozen.py": file_sha256(Path(__file__))},
        "inference": inference,
        **fit_report(records),
    }


def cal_logits(
    checkpoint: Path,
    rows: list[dict[str, Any]],
    max_length: int,
    batch_size: int,
    source_path: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch

    from training.model.decision_model import DecisionModel, collate, encode

    device = torch.device("cuda:0")
    model, tokenizer = DecisionModel.from_checkpoint(
        checkpoint, source_path=source_path
    )
    model = model.float().to(device).eval()
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    encoded = [encode(row, tokenizer, max_length) for row in rows]
    records = []
    with torch.inference_mode():
        for start in range(0, len(encoded), batch_size):
            items = encoded[start : start + batch_size]
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
    import transformers

    runtime = {
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "transformers": transformers.__version__,
    }
    return records, runtime


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cal", type=Path, required=True)
    parser.add_argument("--cal-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--logits", type=Path)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--source-path", type=Path)
    parser.add_argument("--require-kernels", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or (args.logits and args.logits.exists()):
        raise FileExistsError("refusing to overwrite calibration outputs")
    if file_sha256(args.cal) != args.cal_sha256:
        raise ValueError("CAL file differs from its pinned SHA-256")
    kernel_runtime = None
    if args.require_kernels:
        from v2.dec.runtime_check import require_runtime

        kernel_runtime = require_runtime()

    from training.model.data import load_partition
    from training.model.infer import checkpoint_fingerprint

    rows = load_partition(args.cal, "cal")
    identity = checkpoint_fingerprint(args.checkpoint, args.source_path)
    records, runtime = cal_logits(
        args.checkpoint, rows, args.max_length, args.batch_size, args.source_path
    )
    inference = {
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "device": "cuda:0",
        "execution": "BF16 backbone, FP32 head",
        "runtime": runtime,
    }
    if kernel_runtime is not None:
        inference["kernel_runtime"] = kernel_runtime
    report = build_report(records, identity, args.cal_sha256, inference)
    if args.logits:
        with args.logits.open("x", encoding="utf-8") as stream:
            for record in records:
                stream.write(canonical(record) + "\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pending = args.output.with_name(args.output.name + ".pending")
    with pending.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, args.output)
    print(json.dumps({"temperature_by_type": report["temperature_by_type"]}))


if __name__ == "__main__":
    main()
