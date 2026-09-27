"""Gold-free 2B zero-step causal preflight: backend by LoRA attachment.

Each invocation runs one prospectively locked cell. It never scores a label,
updates a weight, or changes the historical control. Four independent cells
are compared only after all their raw predictions have been sealed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch

from .data import file_sha256, load_partition
from .decision_model import DecisionModel
from .inline_replay import roster_sha256
from .inline_teacher import (
    native_probabilities_batch,
    source_pairs_for_selection,
    verify_materialization,
)
from .lora import attach_lora
from .source import source_fingerprint
from .zero_step_repeat import _prediction, compare_historical

SCHEMA = "decision2-sol2b-zero-step-factorial/1"
CELLS = ("fla-bare", "fla-lora", "reference-bare", "reference-lora")
MAX_TOTAL_GPU_SECONDS = 180
MAX_CELL_SECONDS = 45
HISTORICAL_GATE = 1e-4
SAME_RUNTIME_GATE = 1e-4
REPEAT_GATE = 1e-6


def _write_once(path: Path, value: dict) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _lock(path: Path) -> dict:
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("schema_version") != SCHEMA:
        raise ValueError("Wrong preregistration schema")
    if lock.get("runner_sha256") != file_sha256(Path(__file__)):
        raise ValueError("Runner differs from preregistration")
    if (
        lock.get("cells") != list(CELLS)
        or lock.get("max_length") != 8192
        or lock.get("optimizer_updates") != 0
        or lock.get("sample_count") != 32
        or lock.get("max_total_gpu_seconds") != MAX_TOTAL_GPU_SECONDS
        or lock.get("max_cell_seconds") != MAX_CELL_SECONDS
        or lock.get("historical_gate") != HISTORICAL_GATE
        or lock.get("same_runtime_gate") != SAME_RUNTIME_GATE
        or lock.get("repeat_gate") != REPEAT_GATE
    ):
        raise ValueError("Frozen factorial protocol differs")
    for role in ("select", "control", "materialization_receipt"):
        if file_sha256(Path(lock[f"{role}_path"])) != lock[f"{role}_sha256"]:
            raise ValueError(f"{role} differs from preregistration")
    return lock


def _fla_backend() -> dict[str, str]:
    """Confirm the unchanged installed FLA wrapper before model loading."""
    import inspect

    from transformers.models.qwen3_5 import modeling_qwen3_5

    selected = modeling_qwen3_5.torch_chunk_gated_delta_rule
    closure = inspect.getclosurevars(selected).nonlocals
    implementation = closure.get("implementation")
    if (
        closure.get("is_new_implementation") is not True
        or getattr(implementation, "__module__", None)
        != "fla.ops.gated_delta_rule.chunk"
        or getattr(implementation, "__name__", None) != "chunk_gated_delta_rule"
    ):
        raise ValueError("Installed Qwen3.5 gated-delta backend is not pinned FLA")
    return {
        "gated_delta_backend": "fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule"
    }


def _zero_lora(model: DecisionModel, lock: dict) -> int:
    source = source_fingerprint(Path(lock["model_path"]))
    attach_lora(
        model,
        rank=16,
        alpha=32,
        dropout=0.05,
        source_kind="decision2",
        source_fingerprint=source,
    )
    weights = [
        value for name, value in model.backbone.named_parameters() if "lora_B" in name
    ]
    if not weights or any(torch.count_nonzero(value).item() != 0 for value in weights):
        raise ValueError("Fresh LoRA B matrices are not all zero")
    return len(weights)


def _controls(path: Path) -> dict[str, dict]:
    controls = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if item["id"] in controls:
            raise ValueError("Duplicate historical control ID")
        controls[item["id"]] = item
    if len(controls) != 700:
        raise ValueError("Historical SELECT control must have 700 records")
    return controls


def run(lock_path: Path, cell: str, output: Path) -> dict:
    started = time.monotonic()
    lock = _lock(lock_path)
    if cell not in CELLS:
        raise ValueError("Unregistered cell")
    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Exactly one BF16 GPU must be visible")
    torch.manual_seed(20260926)
    torch.cuda.manual_seed_all(20260926)
    torch.use_deterministic_algorithms(True)
    if cell.startswith("reference"):
        from training.eikos.published_infer import use_torch_reference_gated_delta

        backend = use_torch_reference_gated_delta()
    else:
        backend = _fla_backend()
    _, model_sha = verify_materialization(
        Path(lock["model_path"]),
        Path(lock["materialization_receipt_path"]),
        lock["materialization_receipt_sha256"],
    )
    if model_sha != lock["source_model_sha256"]:
        raise ValueError("Source model differs from preregistration")
    rows = load_partition(Path(lock["select_path"]), "select")
    chosen = sorted(
        rows, key=lambda row: hashlib.sha256(row["id"].encode()).hexdigest()
    )[:32]
    if len(rows) != 700 or roster_sha256(chosen) != lock["roster_sha256"]:
        raise ValueError("SELECT or fixed 32-item roster differs")
    selected = {row["id"] for row in chosen}
    model, tokenizer = DecisionModel.from_checkpoint(Path(lock["model_path"]))
    lora_b_count = _zero_lora(model, lock) if cell.endswith("lora") else 0
    model = model.float().to(torch.device("cuda:0"))
    if cell.endswith("lora"):
        model.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model.eval()
    model.backbone.config.use_cache = False
    predictions = []
    with torch.inference_mode():
        for pair in source_pairs_for_selection(rows, selected):
            actuals = native_probabilities_batch(model, tokenizer, pair, 8192)
            for row, (item, probabilities) in zip(pair, actuals):
                if row["id"] in selected:
                    predictions.append(
                        {
                            "id": row["id"],
                            "prompt_sha256": item["prompt_sha256"],
                            "token_ids_sha256": item["token_ids_sha256"],
                            "task_type": row["task_type"],
                            "prediction_key": _prediction(
                                probabilities, row["task_type"]
                            ),
                            "probabilities": probabilities,
                        }
                    )
    if len(predictions) != 32 or any(
        prediction["prediction_key"] is None for prediction in predictions
    ):
        raise ValueError("Incomplete or invalid fixed 32-item predictions")
    historical = compare_historical(predictions, _controls(Path(lock["control_path"])))
    elapsed = time.monotonic() - started
    if elapsed > MAX_CELL_SECONDS:
        raise TimeoutError("Cell exceeded prospective 45-second cap")
    document = {
        "schema_version": SCHEMA,
        "lock_sha256": file_sha256(lock_path),
        "cell": cell,
        "source_model_sha256": model_sha,
        "backend": backend,
        "lora_b_matrix_count": lora_b_count,
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "historical": historical,
        "elapsed_process_seconds": elapsed,
        "predictions": predictions,
    }
    _write_once(output, document)
    return document


def compare_cells(cells: dict[str, dict]) -> dict:
    if set(cells) != set(CELLS):
        raise ValueError("All four frozen cells are required")
    lock_hashes = {item["lock_sha256"] for item in cells.values()}
    source_hashes = {item["source_model_sha256"] for item in cells.values()}
    if len(lock_hashes) != 1 or len(source_hashes) != 1:
        raise ValueError("Factorial cells have different source or protocol")
    by_cell = {}
    for name, item in cells.items():
        if item["cell"] != name:
            raise ValueError("Factorial cell identity differs")
        records = {row["id"]: row for row in item["predictions"]}
        if len(records) != 32:
            raise ValueError("Factorial cell has incomplete roster")
        by_cell[name] = records
    comparisons = {}
    for axis, left, right in (
        ("lora_effect_fla", "fla-bare", "fla-lora"),
        ("lora_effect_reference", "reference-bare", "reference-lora"),
        ("backend_effect_bare", "fla-bare", "reference-bare"),
        ("backend_effect_lora", "fla-lora", "reference-lora"),
    ):
        if by_cell[left].keys() != by_cell[right].keys():
            raise ValueError("Factorial roster IDs differ")
        maximum, changes, rows_over_gate = 0.0, 0, 0
        for identifier in by_cell[left]:
            a, b = by_cell[left][identifier], by_cell[right][identifier]
            if (
                any(
                    a[key] != b[key]
                    for key in ("prompt_sha256", "token_ids_sha256", "task_type")
                )
                or a["probabilities"].keys() != b["probabilities"].keys()
            ):
                raise ValueError("Factorial native input or option keys differ")
            drift = max(
                abs(a["probabilities"][key] - b["probabilities"][key])
                for key in a["probabilities"]
            )
            maximum = max(maximum, drift)
            rows_over_gate += drift > SAME_RUNTIME_GATE
            changes += a["prediction_key"] != b["prediction_key"]
        if not math.isfinite(maximum):
            raise ValueError("Nonfinite factorial probability drift")
        comparisons[axis] = {
            "maximum_probability_drift": maximum,
            "categorical_changes": changes,
            "rows_over_1e-4": rows_over_gate,
            "within_1e-4": changes == 0 and maximum <= SAME_RUNTIME_GATE,
        }
    return {
        "schema_version": SCHEMA,
        "lock_sha256": lock_hashes.pop(),
        "historical": {name: cells[name]["historical"] for name in CELLS},
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--cell", choices=CELLS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.lock, args.cell, args.output)
    print(
        json.dumps(
            {
                "cell": args.cell,
                "historical": result["historical"],
                "prediction_sha256": file_sha256(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
