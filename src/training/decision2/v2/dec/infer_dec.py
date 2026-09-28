"""Gold-blind panel inference for decoder-track checkpoints.

Reuses the shared adapter's prompt contract, answer normalization and sealed
output writer; only model loading and identity differ, so residual readouts
are included in both the forward pass and ``model_sha256``. For checkpoints
without a residual readout the identity equals the shared adapter's.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

from training.model.calibration import load_calibration
from training.model.data import canonical, file_sha256
from training.model.infer import (
    ADAPTER_VERSION,
    CALIBRATED_ADAPTER_VERSION,
    load_prompts,
    run_prompts,
    write_output,
)

SHARED = (
    "infer.py",
    "decision_model.py",
    "data.py",
    "lora.py",
    "source.py",
    "calibration.py",
)
OWN = ("infer_dec.py", "dec_model.py")


def adapter_sources() -> dict[str, str]:
    here = Path(__file__).resolve()
    shared = here.parents[2] / "training" / "model"
    sources = {f"training/model/{name}": file_sha256(shared / name) for name in SHARED}
    sources.update(
        {f"v2/dec/{name}": file_sha256(here.with_name(name)) for name in OWN}
    )
    return sources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument(
        "--max-items", type=int, help="Smoke run on the first N prompts only"
    )
    args = parser.parse_args()
    if (
        args.output.exists()
        or args.output.with_name(args.output.name + ".manifest.json").exists()
    ):
        raise FileExistsError("Refusing to overwrite predictions or manifest")
    rows = load_prompts(args.input)
    if args.max_items is not None:
        rows = rows[: args.max_items]

    import torch

    from training.model.decision_model import collate, encode

    from .dec_model import dec_fingerprint, load_dec_checkpoint
    from .runtime_check import require_runtime

    runtime = require_runtime()
    identity = dec_fingerprint(args.checkpoint, args.source_path)
    calibration = report = None
    if args.calibration is not None:
        calibration, report = load_calibration(
            args.calibration, identity["model_sha256"]
        )
    sources = adapter_sources()
    adapter_sha = hashlib.sha256(canonical(sources).encode("utf-8")).hexdigest()
    device = torch.device("cuda:0")
    model, tokenizer = load_dec_checkpoint(args.checkpoint, args.source_path)
    model = model.float().to(device).eval()
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )

    def predict(encoded: list[dict[str, Any]]) -> list[list[float]]:
        batch = {
            key: (
                value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            )
            for key, value in collate(encoded, pad_id).items()
        }
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            logits = model(**batch)
        return [
            values[: len(item["keys"])].float().cpu().tolist()
            for values, item in zip(logits, encoded)
        ]

    predictions, counts = run_prompts(
        rows,
        tokenizer=tokenizer,
        max_length=args.max_length,
        temperature=calibration if calibration is not None else 1.0,
        encode_fn=encode,
        predict_fn=predict,
        model_sha256=identity["model_sha256"],
        adapter_sha256=adapter_sha,
        calibration_sha256=file_sha256(args.calibration) if args.calibration else None,
    )
    metadata = json.loads(
        (args.checkpoint / "decision_config.json").read_text(encoding="utf-8")
    )
    manifest = {
        "adapter_version": (
            CALIBRATED_ADAPTER_VERSION if calibration else ADAPTER_VERSION
        ),
        "dec_loader": "v2.dec.infer_dec",
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "model_sha256": identity["model_sha256"],
        "model_files_sha256": identity["files_sha256"],
        "checkpoint_format": metadata.get("checkpoint_format", "full"),
        "dec_residual": metadata.get("dec_residual"),
        "peft_version": version("peft"),
        "adapter_sha256": adapter_sha,
        "adapter_files_sha256": sources,
        "input_sha256": file_sha256(args.input),
        "input_items": len(rows),
        "max_length": args.max_length,
        "temperature": 1.0,
        "execution": "one benchmark item at a time; all its questions batched together; BF16 backbone, FP32 head",
        "truncation_policy": "none; over-budget questions produce an invalid answer",
        "counts": counts,
        "torch_version": torch.__version__,
        "runtime": runtime,
    }
    if calibration is not None:
        manifest["calibration"] = {
            "file_sha256": file_sha256(args.calibration),
            "cal_sha256": report["cal_sha256"],
            "temperature_by_type": calibration,
            "binding": "direct",
        }
    write_output(args.output, predictions, manifest)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "counts": counts,
                "model_sha256": identity["model_sha256"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
