"""Fit per-type CAL temperatures for a completed decoder-track run's frozen BEST.

Same fitting code, binding rules and report format as the shared
``training.model.calibrate``; loading and identity include residual readouts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from training.model.calibrate import selected_run
from training.model.calibration import CALIBRATION_VERSION, fit_report
from training.model.data import canonical, file_sha256, load_partition


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cal", type=Path, required=True)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    selected = selected_run(args.run_dir, args.cal)
    cal_rows = load_partition(args.cal, "cal")
    if len(cal_rows) != selected["cal_count"]:
        raise ValueError("CAL row count differs from the audited training run")

    import torch

    from training.model.decision_model import collate, encode

    from .dec_model import dec_fingerprint, load_dec_checkpoint
    from .runtime_check import require_runtime

    runtime = require_runtime()
    identity = dec_fingerprint(selected["checkpoint"], args.source_path)
    checkpoint_format = json.loads(
        (Path(selected["checkpoint"]) / "decision_config.json").read_text()
    ).get("checkpoint_format")
    loaded_source = {
        name.removeprefix("source/"): sha
        for name, sha in identity["files_sha256"].items()
        if name.startswith("source/")
    }
    # A full checkpoint carries every weight itself; its start stays recorded
    # in the run provenance (initialization_source_sha256).
    if (
        checkpoint_format != "full"
        and _digest(loaded_source) != selected["initialization_source_sha256"]
    ):
        raise ValueError(
            "Source files differ from the model source audited in provenance"
        )
    checkpoint_files = {
        name.removeprefix("checkpoint/"): sha
        for name, sha in identity["files_sha256"].items()
        if not name.startswith("source/")
    }
    device = torch.device("cuda:0")
    model, tokenizer = load_dec_checkpoint(selected["checkpoint"], args.source_path)
    model = model.float().to(device).eval()
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    max_length = selected["contract"]["max_length"]
    encoded = [encode(row, tokenizer, max_length) for row in cal_rows]
    records = []
    with torch.inference_mode():
        for start in range(0, len(encoded), args.batch_size):
            items = encoded[start : start + args.batch_size]
            batch = {
                key: (
                    value.to(device, non_blocking=True)
                    if torch.is_tensor(value)
                    else value
                )
                for key, value in collate(items, pad_id).items()
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
    if {r["id"]: (r["task_type"], r["label"]) for r in records} != {
        row["id"]: (row["task_type"], row["label"]) for row in cal_rows
    }:
        raise ValueError("CAL logit collection dropped or changed audited rows")
    report = {
        "calibration_version": CALIBRATION_VERSION,
        "fit_split": "cal",
        "selection_policy": "completed_run_best_only",
        "selected_checkpoint": selected["name"],
        "model_sha256": identity["model_sha256"],
        "checkpoint_sha256": _digest(checkpoint_files),
        "initialization_source_sha256": selected["initialization_source_sha256"],
        "loaded_source_sha256": _digest(loaded_source),
        "cal_sha256": selected["cal_sha256"],
        "best_sha256": selected["best_sha256"],
        "complete_sha256": selected["complete_sha256"],
        "provenance_sha256": selected["provenance_sha256"],
        "logits_sha256": _digest(records),
        "code_sha256": {
            name: file_sha256(Path(__file__).with_name(name))
            for name in ("calibrate_dec.py", "dec_model.py")
        },
        "inference": {
            "max_length": max_length,
            "batch_size": args.batch_size,
            "device": "cuda:0",
            "runtime": runtime,
        },
        **fit_report(records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pending = args.output.with_name(args.output.name + ".pending")
    with pending.open("x", encoding="utf-8") as stream:
        json.dump(
            report,
            stream,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "temperature_by_type": report["temperature_by_type"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
