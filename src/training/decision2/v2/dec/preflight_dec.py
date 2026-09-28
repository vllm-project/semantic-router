"""Zero-step parity and one-step reload gates for one decoder-track arm.

Inputs are two throwaway trainer runs of the same arm configuration: one with
``--zero-step-only`` and one with ``--max-steps 1``. SELECT700 is recomputed
with the trainer's batching.

Code-path identity is tested inside one process, where the runtime is
deterministic: the untouched Decision 1.0 source and the reloaded zero-step
checkpoint must agree exactly, and the reloaded one-step checkpoint must differ
from the source. Comparisons against the trainer's own outputs cross a process
boundary; concurrent jobs can autotune different Triton kernels there, so they
use the measured-noise tolerance below. Reloaded adapter, head and residual
tensors must equal the saved files bit for bit.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from training.model.data import file_sha256, load_partition
from training.model.decision_model import DecisionModel, collate, encode
from training.model.train import atomic_json

from .dec_model import RESIDUAL_FILE, dec_fingerprint, load_dec_checkpoint

EXACT_TOLERANCE = 1e-6
# Concurrent zero-step runs of one model measured max drift .027 with 5/700
# argmax flips (Triton autotuning under contention); in-process runs were exact.
CROSS_PROCESS_DRIFT = 0.05
CROSS_PROCESS_MIN_SAME = 693


def select_probabilities(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    batch_size: int,
    max_length: int,
) -> list[list[float]]:
    device = torch.device("cuda:0")
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    encoded = [encode(row, tokenizer, max_length) for row in rows]
    output: list[list[float]] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(encoded), batch_size):
            items = encoded[start : start + batch_size]
            batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in collate(items, pad_id).items()
            }
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(**batch)
            for item, values in zip(items, logits.float().softmax(-1).cpu().tolist()):
                output.append(values[: len(item["keys"])])
    return output


def stored_probabilities(path: Path, rows: list[dict[str, Any]]) -> list[list[float]]:
    by_id = {
        json.loads(line)["id"]: json.loads(line) for line in path.open(encoding="utf-8")
    }
    result = []
    for row in rows:
        answer = by_id[row["id"]]["answer"]
        keys = [option["key"] for option in row["options"]]
        if answer["type"] == "noul":
            p_true = answer["noul"]
            result.append([1 - p_true if key == "false" else p_true for key in keys])
        else:
            result.append([answer["probabilities"][key] for key in keys])
    return result


def compare(left: list[list[float]], right: list[list[float]]) -> dict[str, Any]:
    drift = max(abs(a - b) for x, y in zip(left, right) for a, b in zip(x, y))
    same = sum(
        max(range(len(x)), key=x.__getitem__) == max(range(len(y)), key=y.__getitem__)
        for x, y in zip(left, right)
    )
    return {"n": len(left), "same_argmax": same, "max_probability_drift": drift}


def tensors_match_files(model: Any, checkpoint: Path) -> dict[str, Any]:
    from safetensors.torch import load_file

    adapter = load_file(str(checkpoint / "adapter" / "adapter_model.safetensors"))
    live = {
        name.replace(".default", ""): p.detach().cpu()
        for name, p in model.backbone.named_parameters()
        if "lora_" in name
    }
    adapter_equal = set(adapter) == set(live) and all(
        torch.equal(adapter[k], live[k].float()) for k in adapter
    )
    head_file = load_file(str(checkpoint / "decision_head.safetensors"))
    head_live = {k: v.detach().cpu() for k, v in model.head.state_dict().items()}
    head_equal = set(head_file) == set(head_live) and all(
        torch.equal(head_file[k], head_live[k].float()) for k in head_file
    )
    residual_equal = True
    if (checkpoint / RESIDUAL_FILE).is_file():
        residual_file = load_file(str(checkpoint / RESIDUAL_FILE))
        residual_live = model.residual_state()
        residual_equal = set(residual_file) == set(residual_live) and all(
            torch.equal(residual_file[k], residual_live[k]) for k in residual_file
        )
    return {
        "adapter_equal": adapter_equal,
        "head_equal": head_equal,
        "residual_equal": residual_equal,
        "lora_b_abs_sum": sum(
            v.abs().sum().item() for k, v in live.items() if "lora_B" in k
        ),
    }


def cross_ok(result: dict[str, Any]) -> bool:
    return (
        result["same_argmax"] >= CROSS_PROCESS_MIN_SAME
        and result["max_probability_drift"] <= CROSS_PROCESS_DRIFT
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--select", type=Path, required=True)
    parser.add_argument("--zero-run", type=Path, required=True)
    parser.add_argument("--one-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = load_partition(args.select, "select")
    ignored = ("zero_step_only", "max_steps", "planned_updates", "checkpoint_steps")
    contract = json.loads((args.zero_run / "provenance.json").read_text())["contract"]
    one_contract = json.loads((args.one_run / "provenance.json").read_text())[
        "contract"
    ]
    if {k: v for k, v in contract.items() if k not in ignored} != {
        k: v for k, v in one_contract.items() if k not in ignored
    }:
        raise ValueError("Zero-step and one-step runs differ in arm configuration")
    batch, max_length = contract["eval_batch"], contract["max_length"]
    device = torch.device("cuda:0")
    checks: dict[str, Any] = {
        "arm": contract["arm"],
        "select_sha256": file_sha256(args.select),
    }

    reference, tokenizer = DecisionModel.from_decision1(
        args.source_path, contract["head_dim"]
    )
    source_head = {
        k: v.detach().clone() for k, v in reference.head.state_dict().items()
    }
    reference = reference.float().to(device)
    ref_probs = select_probabilities(reference, tokenizer, rows, batch, max_length)
    del reference
    torch.cuda.empty_cache()

    zero_ckpt = args.zero_run / "checkpoint-0000000"
    zero_model, zero_tok = load_dec_checkpoint(zero_ckpt, args.source_path)
    zero_model = zero_model.float().to(device)
    checks["zero_source_vs_reload_in_process"] = compare(
        ref_probs, select_probabilities(zero_model, zero_tok, rows, batch, max_length)
    )
    del zero_model
    torch.cuda.empty_cache()
    checks["zero_source_vs_trainer_cross_process"] = compare(
        ref_probs,
        stored_probabilities(args.zero_run / "select-baseline-predictions.jsonl", rows),
    )

    one_ckpt = args.one_run / "checkpoint-0000001"
    one_model, one_tok = load_dec_checkpoint(one_ckpt, args.source_path)
    one_model = one_model.float().to(device)
    one_probs = select_probabilities(one_model, one_tok, rows, batch, max_length)
    checks["one_step_tensors"] = tensors_match_files(one_model, one_ckpt)
    checks["one_step_head_max_change"] = max(
        (one_model.head.state_dict()[k].detach().cpu() - v).abs().max().item()
        for k, v in source_head.items()
    )
    checks["one_step_gates"] = {
        name: getattr(one_model, name).gate.item()
        for name in ("ordinal_score", "layer_mix")
        if getattr(one_model, name) is not None
    }
    del one_model
    torch.cuda.empty_cache()
    checks["one_step_source_vs_reload_in_process"] = compare(ref_probs, one_probs)
    checks["one_step_trainer_vs_reload_cross_process"] = compare(
        stored_probabilities(
            args.one_run / "select-step-0000001-predictions.jsonl", rows
        ),
        one_probs,
    )
    events = [
        json.loads(line)
        for line in (args.one_run / "train-metrics.jsonl").open()
        if '"event": "train"' in line
    ]
    checks["one_step_train_event"] = events[0] if events else None
    checks["zero_identity"] = dec_fingerprint(zero_ckpt, args.source_path)[
        "model_sha256"
    ]
    checks["one_identity"] = dec_fingerprint(one_ckpt, args.source_path)["model_sha256"]

    exact = checks["zero_source_vs_reload_in_process"]
    tensors = checks["one_step_tensors"]
    gates = {
        "zero_reload_exact_in_process": exact["same_argmax"] == len(rows)
        and exact["max_probability_drift"] <= EXACT_TOLERANCE,
        "zero_trainer_cross_process": cross_ok(
            checks["zero_source_vs_trainer_cross_process"]
        ),
        "one_step_reload_cross_process": cross_ok(
            checks["one_step_trainer_vs_reload_cross_process"]
        ),
        "one_step_active_in_process": checks["one_step_source_vs_reload_in_process"][
            "max_probability_drift"
        ]
        > 0,
        "one_step_tensors_reload_bitwise": tensors["adapter_equal"]
        and tensors["head_equal"]
        and tensors["residual_equal"],
        "one_step_moved": tensors["lora_b_abs_sum"] > 0
        and checks["one_step_head_max_change"] > 0
        and all(v != 0 for v in checks["one_step_gates"].values()),
        "one_step_finite": bool(events)
        and all(math.isfinite(events[0][k]) for k in ("loss", "gradient_norm")),
    }
    receipt = {
        "schema_version": "dec-arm-preflight/2",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "tolerances": {
            "exact_in_process": EXACT_TOLERANCE,
            "cross_process_drift": CROSS_PROCESS_DRIFT,
            "cross_process_min_same_argmax": CROSS_PROCESS_MIN_SAME,
        },
        "checks": checks,
        "code_sha256": {
            name: file_sha256(Path(__file__).with_name(name))
            for name in ("preflight_dec.py", "dec_model.py", "train_dec.py")
        },
    }
    atomic_json(args.output, receipt)
    print(json.dumps({"status": receipt["status"], "gates": gates}), flush=True)


if __name__ == "__main__":
    main()
