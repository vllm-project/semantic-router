"""Score gold-free benchmark prompts with a selected Eikos native-letter LoRA."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from inference.eikos import REVISION, completed_rows, shared_answer
from inference.run import digest, file_digest, load_prompts, synchronize

from training.eikos.native import load_decider, selected_checkpoint

ADAPTER_VERSION = "eikos-native-letter-lora-v1"


def collect(
    *,
    model_path: Path,
    run: Path,
    prompts: Path,
    output: Path,
    calibration: Path | None = None,
    device: str = "cuda:0",
    resume: bool = False,
    max_items: int | None = None,
) -> dict:
    if max_items is not None and max_items < 1:
        raise ValueError("max_items must be positive")
    selection = selected_checkpoint(run, model_path)
    calibration = calibration or model_path / "calib.json"
    calibration_sha = file_digest(calibration)
    rows = load_prompts(prompts)
    identity = {
        "backend": "eikos_lora",
        "model_id": "caiovicentino1/Eikos-4B+selected-LoRA",
        "model_revision": REVISION,
        "adapter_version": ADAPTER_VERSION,
        "source_release_manifest_sha256": selection["source_release"][
            "release_manifest_sha256"
        ],
        "training_provenance_sha256": selection["provenance_sha256"],
        "adapter_weights_sha256": selection["adapter_weights_sha256"],
        "adapter_config_sha256": selection["adapter_config_sha256"],
        "calibration_sha256": calibration_sha,
        "checkpoint": selection["name"],
    }
    if output.exists():
        if not resume:
            raise FileExistsError(output)
        seen = completed_rows(output, rows, identity)
    else:
        seen = set()
    pending = [row for row in rows if row["id"] not in seen]
    if max_items is not None:
        pending = pending[:max_items]
    native = load_decider(model_path, selection["adapter"], calibration, device=device)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a" if output.exists() else "x", encoding="utf-8") as stream:
        for row in pending:
            payload = {"state": row["state"], "questions": row["questions"]}
            synchronize(device)
            started = time.perf_counter()
            try:
                native_result = native.decide_all(**payload)
                answers = {
                    key: shared_answer(row["questions"][key], answer)
                    for key, (answer, _) in native_result.items()
                }
                usage = {
                    "input_tokens": sum(count for _, count in native_result.values()),
                    "output_tokens": 0,
                }
                invalid_reason = None
            except ValueError as exc:
                if "tokens > 16000" not in str(exc):
                    raise
                answers = {
                    key: {"type": question["type"], "error": "context_overflow"}
                    for key, question in row["questions"].items()
                }
                usage = None
                invalid_reason = "context_overflow"
            synchronize(device)
            latency_ms = 1000 * (time.perf_counter() - started)
            if set(answers) != set(row["questions"]) or not math.isfinite(latency_ms):
                raise ValueError(f"{row['id']}: incomplete native answer")
            receipt = {
                "id": row["id"],
                "answers": answers,
                "usage": usage,
                "latency_ms": latency_ms,
                "source_input_sha256": digest(payload),
                "model": f"Eikos-4B@{REVISION}+{selection['name']}",
                "runtime_qualification": "pytorch_bf16_rocm_unvalidated",
                "invalid_reason": invalid_reason,
                **identity,
            }
            stream.write(
                json.dumps(
                    receipt, ensure_ascii=False, separators=(",", ":"), allow_nan=False
                )
                + "\n"
            )
            stream.flush()
    return {
        "input_items": len(rows),
        "previously_completed": len(seen),
        "collected_now": len(pending),
        "output": str(output),
        **identity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-items", type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            collect(
                model_path=args.model_path,
                run=args.run,
                prompts=args.input,
                output=args.output,
                calibration=args.calibration,
                device=args.device,
                resume=args.resume,
                max_items=args.max_items,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
