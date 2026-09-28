"""Zero-step parity and one-step reload gates for one decoder-track arm.

Inputs are two throwaway trainer runs of the same arm configuration: one with
``--zero-step-only`` and one with ``--max-steps 1``. The gate recomputes the
full SELECT panel with the trainer's batching and requires:

1. the untouched Decision 1.0 source, the in-memory zero-step arm model and its
   reloaded zero-step checkpoint to agree (argmax identical, probability drift
   at most ``ZERO_TOLERANCE``);
2. the reloaded one-step checkpoint to reproduce the trainer's in-memory
   post-update SELECT outputs (argmax identical, drift at most ``RELOAD_TOLERANCE``);
3. the update to be finite and to move LoRA, head and any residual gate.
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

from .dec_model import dec_fingerprint, load_dec_checkpoint

ZERO_TOLERANCE = 1e-5
RELOAD_TOLERANCE = 1e-4


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
    by_id = {}
    for line in path.open(encoding="utf-8"):
        record = json.loads(line)
        by_id[record["id"]] = record
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--select", type=Path, required=True)
    parser.add_argument("--zero-run", type=Path, required=True)
    parser.add_argument("--one-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = load_partition(args.select, "select")
    provenance = json.loads((args.zero_run / "provenance.json").read_text())
    contract = provenance["contract"]
    one_contract = json.loads((args.one_run / "provenance.json").read_text())[
        "contract"
    ]
    comparable = {
        k: v
        for k, v in contract.items()
        if k not in ("zero_step_only", "max_steps", "planned_updates")
    }
    if comparable != {
        k: v
        for k, v in one_contract.items()
        if k not in ("zero_step_only", "max_steps", "planned_updates")
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
    reference = reference.float().to(device)
    ref_probs = select_probabilities(reference, tokenizer, rows, batch, max_length)
    del reference
    torch.cuda.empty_cache()

    zero_ckpt = args.zero_run / "checkpoint-0000000"
    zero_model, zero_tok = load_dec_checkpoint(zero_ckpt, args.source_path)
    zero_model = zero_model.float().to(device)
    zero_probs = select_probabilities(zero_model, zero_tok, rows, batch, max_length)
    del zero_model
    torch.cuda.empty_cache()
    checks["zero_source_vs_trainer"] = compare(
        ref_probs,
        stored_probabilities(args.zero_run / "select-baseline-predictions.jsonl", rows),
    )
    checks["zero_source_vs_reload"] = compare(ref_probs, zero_probs)

    one_ckpt = args.one_run / "checkpoint-0000001"
    one_model, one_tok = load_dec_checkpoint(one_ckpt, args.source_path)
    one_model = one_model.float().to(device)
    one_probs = select_probabilities(one_model, one_tok, rows, batch, max_length)
    moved = {
        "lora_b_abs_sum": sum(
            p.detach().abs().sum().item()
            for n, p in one_model.backbone.named_parameters()
            if "lora_B" in n
        ),
    }
    source_head, _ = DecisionModel.from_decision1(
        args.source_path, contract["head_dim"]
    )
    moved["head_max_abs_change"] = max(
        (a.detach().cpu() - b.detach()).abs().max().item()
        for a, b in zip(
            one_model.head.state_dict().values(), source_head.head.state_dict().values()
        )
    )
    del source_head
    for name in ("ordinal_score", "layer_mix"):
        module = getattr(one_model, name)
        if module is not None:
            moved[f"{name}_gate"] = module.gate.item()
    del one_model
    torch.cuda.empty_cache()
    checks["one_step_trainer_vs_reload"] = compare(
        stored_probabilities(
            args.one_run / "select-step-0000001-predictions.jsonl", rows
        ),
        one_probs,
    )
    checks["one_step_changed_vs_source"] = compare(ref_probs, one_probs)
    checks["moved"] = moved
    train_events = [
        json.loads(line)
        for line in (args.one_run / "train-metrics.jsonl").open()
        if '"event": "train"' in line
    ]
    checks["one_step_train_event"] = train_events[0] if train_events else None
    checks["zero_identity"] = dec_fingerprint(zero_ckpt, args.source_path)[
        "model_sha256"
    ]
    checks["one_identity"] = dec_fingerprint(one_ckpt, args.source_path)["model_sha256"]

    gates = {
        "zero_trainer_parity": checks["zero_source_vs_trainer"]["same_argmax"]
        == len(rows)
        and checks["zero_source_vs_trainer"]["max_probability_drift"] <= ZERO_TOLERANCE,
        "zero_reload_parity": checks["zero_source_vs_reload"]["same_argmax"]
        == len(rows)
        and checks["zero_source_vs_reload"]["max_probability_drift"] <= ZERO_TOLERANCE,
        "one_step_reload_parity": checks["one_step_trainer_vs_reload"]["same_argmax"]
        == len(rows)
        and checks["one_step_trainer_vs_reload"]["max_probability_drift"]
        <= RELOAD_TOLERANCE,
        "one_step_finite": bool(train_events)
        and all(math.isfinite(train_events[0][k]) for k in ("loss", "gradient_norm")),
        "one_step_moved": moved["lora_b_abs_sum"] > 0
        and moved["head_max_abs_change"] > 0
        and all(abs(v) > 0 for k, v in moved.items() if k.endswith("_gate")),
    }
    receipt = {
        "schema_version": "dec-arm-preflight/1",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "tolerances": {"zero": ZERO_TOLERANCE, "reload": RELOAD_TOLERANCE},
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
