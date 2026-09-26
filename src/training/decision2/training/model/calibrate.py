"""Fit post-hoc native-type temperatures on a completed run's audited CAL only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from .calibration import CALIBRATION_VERSION, fit_report
from .data import canonical, file_sha256, load_partition
from .infer import checkpoint_fingerprint


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def selected_run(run_dir: Path, cal_path: Path) -> dict[str, Any]:
    """Bind CAL bytes to the completed run's frozen BEST checkpoint."""
    best_path = run_dir / "BEST.json"
    complete_path = run_dir / "COMPLETE.json"
    provenance_path = run_dir / "provenance.json"
    best = json.loads(best_path.read_text(encoding="utf-8"))
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if any(not isinstance(value, dict) for value in (best, complete, provenance)):
        raise ValueError("Run selection and provenance files must be JSON objects")
    name = best.get("checkpoint")
    if not isinstance(name, str) or re.fullmatch(r"checkpoint-[0-9]{7}", name) is None:
        raise ValueError(
            "BEST.json must name a complete checkpoint directly inside the run"
        )
    if complete.get("status") != "complete" or complete.get("best") != name:
        raise ValueError(
            "Calibration requires a completed run with a frozen BEST checkpoint"
        )
    checkpoint = run_dir / name
    checkpoint_receipt = json.loads(
        (checkpoint / "checkpoint.json").read_text(encoding="utf-8")
    )
    if (
        not isinstance(checkpoint_receipt, dict)
        or checkpoint_receipt.get("complete") is not True
    ):
        raise ValueError("Selected checkpoint receipt is incomplete")
    contract = provenance.get("contract")
    if not isinstance(contract, dict) or not isinstance(
        contract.get("data_sha256"), dict
    ):
        raise ValueError("Run provenance lacks the audited data contract")
    cal_sha = file_sha256(cal_path)
    if contract["data_sha256"].get("cal") != cal_sha:
        raise ValueError(
            "CAL file differs from the partition audited by this training run"
        )
    if (
        type(provenance.get("cal_examples_audited_only")) is not int
        or provenance["cal_examples_audited_only"] < 1
    ):
        raise ValueError("Run provenance does not attest CAL was audit-only")
    source = provenance.get("model_source")
    if not isinstance(source, dict) or not isinstance(source.get("files_sha256"), dict):
        raise ValueError("Run provenance lacks source model file hashes")
    return {
        "checkpoint": checkpoint,
        "name": name,
        "contract": contract,
        "cal_sha256": cal_sha,
        "best_sha256": file_sha256(best_path),
        "complete_sha256": file_sha256(complete_path),
        "provenance_sha256": file_sha256(provenance_path),
        "initialization_source_sha256": _digest(source["files_sha256"]),
        "cal_count": provenance["cal_examples_audited_only"],
    }


def collect_logits(
    checkpoint: Path,
    cal_rows: list[dict[str, Any]],
    *,
    source_path: Path | None,
    max_length: int,
    batch_size: int,
    device_name: str,
) -> list[dict[str, Any]]:
    import torch

    from .decision_model import DecisionModel, collate, encode

    if max_length < 1 or batch_size < 1:
        raise ValueError("max_length and batch_size must be positive")
    device = torch.device(device_name)
    if device.type == "cuda" and (
        not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("Calibration needs a CUDA/ROCm device with BF16 support")
    model, tokenizer = DecisionModel.from_checkpoint(
        checkpoint, source_path=source_path
    )
    model = model.float().to(device).eval()
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    if pad_id is None:
        raise ValueError("Tokenizer needs a pad or EOS token")
    encoded = [encode(row, tokenizer, max_length) for row in cal_rows]
    records = []
    with torch.inference_mode():
        for start in range(0, len(encoded), batch_size):
            items = encoded[start : start + batch_size]
            batch = {
                key: (
                    value.to(device, non_blocking=True)
                    if torch.is_tensor(value)
                    else value
                )
                for key, value in collate(items, pad_id).items()
            }
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device.type == "cuda"
                else nullcontext()
            )
            with autocast:
                logits = model(**batch)
            for item, output in zip(items, logits):
                values = output[: len(item["keys"])].float().cpu().tolist()
                records.append(
                    {
                        "id": item["id"],
                        "task_type": item["task_type"],
                        "label": item["label"],
                        "logits": values,
                    }
                )
    return records


def calibrate(
    run_dir: Path,
    cal_path: Path,
    output: Path,
    *,
    source_path: Path | None = None,
    batch_size: int = 2,
    device_name: str = "cuda:0",
) -> dict[str, Any]:
    if output.exists() or output.with_name(output.name + ".pending").exists():
        raise FileExistsError("Calibration output or pending file already exists")
    selected = selected_run(run_dir, cal_path)
    if output.resolve().is_relative_to(selected["checkpoint"].resolve()):
        raise ValueError(
            "Calibration output cannot be placed inside the hashed checkpoint"
        )
    cal_rows = load_partition(cal_path, "cal")
    if len(cal_rows) != selected["cal_count"]:
        raise ValueError("CAL row count differs from the audited training run")
    identity = checkpoint_fingerprint(selected["checkpoint"], source_path)
    loaded_source_files = {
        name.removeprefix("source/"): sha
        for name, sha in identity["files_sha256"].items()
        if name.startswith("source/")
    }
    if (
        loaded_source_files
        and _digest(loaded_source_files) != selected["initialization_source_sha256"]
    ):
        raise ValueError(
            "LoRA source files differ from the model source audited in run provenance"
        )
    checkpoint_files = {
        name.removeprefix("checkpoint/"): sha
        for name, sha in identity["files_sha256"].items()
        if not name.startswith("source/")
    }
    if not checkpoint_files:
        raise ValueError("Selected checkpoint has no model artifacts")
    records = collect_logits(
        selected["checkpoint"],
        cal_rows,
        source_path=source_path,
        max_length=selected["contract"]["max_length"],
        batch_size=batch_size,
        device_name=device_name,
    )
    expected_rows = {row["id"]: (row["task_type"], row["label"]) for row in cal_rows}
    observed_rows = {row["id"]: (row["task_type"], row["label"]) for row in records}
    if len(records) != len(cal_rows) or observed_rows != expected_rows:
        raise ValueError(
            "CAL logit collection dropped, duplicated, or changed audited rows"
        )
    result = fit_report(records)
    report = {
        "calibration_version": CALIBRATION_VERSION,
        "fit_split": "cal",
        "selection_policy": "completed_run_best_only",
        "selected_checkpoint": selected["name"],
        "model_sha256": identity["model_sha256"],
        "checkpoint_sha256": _digest(checkpoint_files),
        "initialization_source_sha256": selected["initialization_source_sha256"],
        "loaded_source_sha256": (
            _digest(loaded_source_files) if loaded_source_files else None
        ),
        "cal_sha256": selected["cal_sha256"],
        "best_sha256": selected["best_sha256"],
        "complete_sha256": selected["complete_sha256"],
        "provenance_sha256": selected["provenance_sha256"],
        "logits_sha256": _digest(records),
        "code_sha256": {
            name: file_sha256(Path(__file__).with_name(name))
            for name in (
                "calibrate.py",
                "calibration.py",
                "decision_model.py",
                "infer.py",
                "lora.py",
                "source.py",
                "data.py",
            )
        },
        "inference": {
            "max_length": selected["contract"]["max_length"],
            "batch_size": batch_size,
            "device": device_name,
            "precision": "FP32 parameters; BF16 backbone compute and FP32 head on CUDA",
        },
        "metric_policy": {
            "objective": "Unweighted hard-label NLL within each native task type",
            "brier": "Multiclass squared error divided by two, over all CAL rows",
            "ece_10": "Ten equal-width maximum-probability bins, over all CAL rows",
            "missing_policy": "Every audited CAL row must produce finite logits; no truncation or drop",
        },
        **result,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = output.with_name(output.name + ".pending")
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
    if output.exists():
        raise FileExistsError(output)
    os.replace(pending, output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Completed training run containing BEST.json",
    )
    parser.add_argument(
        "--cal", type=Path, required=True, help="Same CAL JSONL audited by that run"
    )
    parser.add_argument(
        "--source-path",
        type=Path,
        help="Required immutable source for a LoRA checkpoint",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    report = calibrate(
        args.run_dir,
        args.cal,
        args.output,
        source_path=args.source_path,
        batch_size=args.batch_size,
        device_name=args.device,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "model_sha256": report["model_sha256"],
                "temperature_by_type": report["temperature_by_type"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
