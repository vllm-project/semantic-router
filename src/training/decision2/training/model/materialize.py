"""Merge an audited LoRA adapter into a portable full Decision 2.0 checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .data import file_sha256
from .infer import checkpoint_fingerprint


def materialize(
    checkpoint: Path, source_path: Path, output: Path, device_name: str = "cpu"
) -> dict[str, Any]:
    if (
        output.exists()
        or output.with_name(output.name + ".materialization.json").exists()
    ):
        raise FileExistsError("Output checkpoint or receipt already exists")
    source_identity = checkpoint_fingerprint(checkpoint, source_path)

    import torch

    from .decision_model import DecisionModel

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA/ROCm device is unavailable")
    model, tokenizer = DecisionModel.from_checkpoint(
        checkpoint, source_path=source_path
    )
    model = model.float().to(device).eval()
    model.merge_lora(source_identity)
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = output.with_name(output.name + ".pending")
    if pending.exists():
        raise FileExistsError(f"Interrupted pending output exists: {pending}")
    model.save(pending, tokenizer)
    merged_identity = checkpoint_fingerprint(pending)
    receipt = {
        "materialization_version": "decision2-merged-peft-lora/1",
        "source_model_sha256": source_identity["model_sha256"],
        "source_model_files_sha256": source_identity["files_sha256"],
        "merged_model_sha256": merged_identity["model_sha256"],
        "merged_model_files_sha256": merged_identity["files_sha256"],
        "materializer_code_sha256": file_sha256(Path(__file__)),
        "precision": "FP32 merged backbone and FP32 decision head",
    }
    receipt_path = output.with_name(output.name + ".materialization.json")
    receipt_pending = receipt_path.with_name(receipt_path.name + ".pending")
    receipt_text = (
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    # The portable copy travels inside a published full-model bundle. The
    # checkpoint fingerprint hashes only inference artifacts, not this receipt.
    with (pending / "materialization_receipt.json").open(
        "x", encoding="utf-8"
    ) as stream:
        stream.write(receipt_text)
        stream.flush()
        os.fsync(stream.fileno())
    with receipt_pending.open("x", encoding="utf-8") as stream:
        stream.write(receipt_text)
        stream.flush()
        os.fsync(stream.fileno())
    if output.exists() or receipt_path.exists():
        raise FileExistsError("Output appeared while materializing")
    os.replace(pending, output)
    os.replace(receipt_pending, receipt_path)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device", default="cpu", help="cpu or a GPU device such as cuda:0"
    )
    args = parser.parse_args()
    receipt = materialize(args.checkpoint, args.source_path, args.output, args.device)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "merged_model_sha256": receipt["merged_model_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
