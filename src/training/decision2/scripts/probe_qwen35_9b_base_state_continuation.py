"""Bounded exact-state technical continuation of the failed 9B Base arm.

Only original TRAIN rows and the saved step-64 state are read. The probe
mutates an in-memory model through update 107 at most, without saving weights
or evaluating any selection, calibration, or final partition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any

from probe_qwen35_9b_base_backward import (
    ACCUMULATION,
    EXPECTED,
    SOURCE_HASHES,
    canonical,
    check_inputs,
    sha256,
)

FIRST_UPDATE = 65
LAST_UPDATE = 107
EXPECTED_MANIFEST = "d1554946856a3c9ee29a22376a61951d6934ad8a2b77d54c9d11900bb28eaa08"


def original_steps(path: Path) -> dict[int, dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    selected = {
        row["step"]: row
        for row in rows
        if row.get("event") == "train" and FIRST_UPDATE <= row["step"] < LAST_UPDATE
    }
    if sorted(selected) != list(range(FIRST_UPDATE, LAST_UPDATE)):
        raise ValueError("Original complete update roster differs")
    return selected


def compare_step(
    step: int, observed: dict[str, float | int], expected: dict[str, Any]
) -> dict[str, float]:
    """Stop on a trajectory mismatch before exercising the faulting window."""
    if observed["tokens"] != expected["tokens"]:
        raise ValueError(f"Update {step}: TRAIN token schedule diverged")
    deviations = {}
    for key, relative_limit in (("loss", 0.01), ("gradient_norm", 0.05)):
        reference = float(expected[key])
        actual = float(observed[key])
        if not all(map(math.isfinite, (reference, actual))):
            raise ValueError(f"Update {step}: nonfinite {key}")
        relative = abs(actual - reference) / max(abs(reference), 1e-6)
        deviations[key] = relative
        if relative > relative_limit:
            raise ValueError(f"Update {step}: {key} trajectory diverged")
    return deviations


class Events:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("x", encoding="utf-8")
        os.chmod(path, 0o600)

    def emit(self, **record: Any) -> None:
        self.stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def close(self) -> None:
        self.stream.close()


def _grad_tensor(value: Any) -> Any | None:
    import torch

    if isinstance(value, torch.Tensor) and value.requires_grad:
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            result = _grad_tensor(item)
            if result is not None:
                return result
    return None


def attach_target_hooks(model: Any, events: Events, active: dict[str, Any]) -> list:
    """Mark the last entered layer component if the target backward crashes."""
    handles = []
    for name, module in model.named_modules():
        if not (
            re.search(r"(?:^|\.)layers\.\d+$", name)
            or name.endswith((".linear_attn", ".self_attn", ".mlp"))
        ):
            continue

        def on_forward(
            _module: Any, _inputs: Any, output: Any, *, label: str = name
        ) -> None:
            if not active["enabled"]:
                return
            tensor = _grad_tensor(output)
            if tensor is None:
                return
            events.emit(
                step=LAST_UPDATE,
                microbatch=active["microbatch"],
                phase="module_forward",
                module=label,
            )

            def on_gradient(gradient: Any) -> Any:
                events.emit(
                    step=LAST_UPDATE,
                    microbatch=active["microbatch"],
                    phase="module_backward_entry",
                    module=label,
                )
                return gradient

            tensor.register_hook(on_gradient)

        handles.append(module.register_forward_hook(on_forward))
    if len(handles) < 32:
        raise ValueError("Expected 9B layer instrumentation was not installed")
    return handles


def read_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    fingerprint = manifest.pop("manifest_sha256")
    if (
        fingerprint != EXPECTED_MANIFEST
        or hashlib.sha256(canonical(manifest)).hexdigest() != fingerprint
    ):
        raise ValueError("Sealed TRAIN schedule manifest differs")
    manifest["manifest_sha256"] = fingerprint
    if (
        manifest["train_sha256"] != EXPECTED["train"]
        or manifest["source_sha256"] != SOURCE_HASHES
    ):
        raise ValueError("Sealed TRAIN or trainer source differs")
    return manifest


def continue_state(args: argparse.Namespace) -> dict[str, Any]:
    from importlib.metadata import version

    import torch
    from training.model.data import load_partition
    from training.model.decision_model import DecisionModel, collate, encode
    from training.model.lora import adapter_parameters
    from training.model.loss import per_example_loss
    from training.model.plan import epoch_batches, validate_resume_state
    from training.model.train import learning_factor

    provenance, checkpoint = check_inputs(args)
    manifest = read_manifest(args.manifest)
    contract = provenance["contract"]
    if (
        contract["choice_source"] != []
        or contract["weighted_choice_count"] != 0
        or contract["replay_fraction"] != 0.0
        or contract["replay_kl_weight"] != 0.0
        or contract["gradient_checkpointing"] is not False
        or contract["lora"]["rank"] != 16
        or contract["lora"]["alpha"] != 32
        or contract["lora"]["dropout"] != 0.05
    ):
        raise ValueError("Original optimizer or sampling contract differs")
    if (
        checkpoint["step"] != FIRST_UPDATE - 1
        or checkpoint["next_batch"] != (FIRST_UPDATE - 1) * ACCUMULATION
    ):
        raise ValueError("Saved data cursor differs")
    if (
        not torch.cuda.is_available()
        or not torch.cuda.is_bf16_supported()
        or torch.cuda.device_count() != 1
    ):
        raise RuntimeError("Exactly one BF16 ROCm accelerator is required")
    if (torch.__version__, torch.version.hip, version("peft")) != (
        provenance["torch_version"],
        provenance["hip_version"],
        contract["lora"]["peft_version"],
    ):
        raise ValueError("Runtime or PEFT version differs from original arm")

    random.seed(contract["seed"])
    torch.manual_seed(contract["seed"])
    torch.cuda.manual_seed_all(contract["seed"])
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")
    model, tokenizer = DecisionModel.from_checkpoint(
        args.run / "checkpoint-0000064", source_path=args.model, trainable_adapter=True
    )
    if model.metadata["lora"]["source_fingerprint"] != provenance["model_source"]:
        raise ValueError("Loaded source differs from original run")
    model = model.float().to(device)
    model.backbone.config.use_cache = False
    model.train()

    rows = load_partition(args.train, "train")
    items = [encode(row, tokenizer, contract["max_length"]) for row in rows]
    lengths = [len(item["ids"]) for item in items]
    if len(items) != 7324 or sum(lengths) != 3579176:
        raise ValueError("Original TRAIN tokenization differs")
    batches = epoch_batches(
        lengths, [], epoch=0, seed=contract["seed"], microbatch=1, replay_fraction=0.0
    )
    if [
        batch[0][1]
        for batch in batches[
            (LAST_UPDATE - 1) * ACCUMULATION : LAST_UPDATE * ACCUMULATION
        ]
    ] != manifest["schedule"][str(LAST_UPDATE)]:
        raise ValueError("Target TRAIN window differs")

    optimizer = torch.optim.AdamW(
        [
            {
                "params": adapter_parameters(model),
                "lr": contract["lora"]["lr"],
                "peak_lr": contract["lora"]["lr"],
                "name": "lora",
            },
            {
                "params": list(model.head.parameters()),
                "lr": contract["head_lr"],
                "peak_lr": contract["head_lr"],
                "name": "head",
            },
        ],
        weight_decay=contract["weight_decay"],
        foreach=True,
    )
    state = torch.load(
        args.run / "checkpoint-0000064/trainer_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    validate_resume_state(state, contract, SOURCE_HASHES)
    optimizer.load_state_dict(state["optimizer"])
    random.setstate(state["python_rng"])
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state(state["cuda_rng"], device)
    del state
    pad = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    if pad is None:
        raise ValueError("No tokenizer pad or EOS ID")
    expected_steps = original_steps(args.run / "train-metrics.jsonl")
    events = Events(args.events)
    active: dict[str, Any] = {"enabled": False, "microbatch": -1}
    handles = attach_target_hooks(model, events, active)
    largest_deviation = {"loss": 0.0, "gradient_norm": 0.0}
    try:
        events.emit(
            phase="start", checkpoint_step=64, manifest_sha256=EXPECTED_MANIFEST
        )
        for step in range(FIRST_UPDATE, LAST_UPDATE + 1):
            window = batches[(step - 1) * ACCUMULATION : step * ACCUMULATION]
            if len(window) != ACCUMULATION or any(
                len(batch) != 1 or batch[0][0] != "train" for batch in window
            ):
                raise ValueError("Unexpected original TRAIN batch shape")
            factor = learning_factor(
                step - 1, contract["planned_updates"], contract["warmup_ratio"]
            )
            for group in optimizer.param_groups:
                group["lr"] = group["peak_lr"] * factor
            optimizer.zero_grad(set_to_none=True)
            weighted_total = 0.0
            tokens = 0
            for microbatch, batch_ids in enumerate(window):
                row_index = batch_ids[0][1]
                item = items[row_index]
                batch = {
                    key: (
                        value.to(device, non_blocking=True)
                        if torch.is_tensor(value)
                        else value
                    )
                    for key, value in collate([item], pad).items()
                }
                if step == LAST_UPDATE:
                    evidence = manifest["rows"][str(row_index)]
                    if item["token_ids_sha256"] != evidence["token_ids_sha256"]:
                        raise ValueError("Target row tokens differ")
                    active.update(enabled=True, microbatch=microbatch)
                    events.emit(
                        step=step,
                        microbatch=microbatch,
                        phase="forward_start",
                        row_sha256=evidence["row_sha256"],
                        tokens=len(item["ids"]),
                    )
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(**batch)
                    terms = per_example_loss(
                        logits,
                        batch["labels"],
                        batch["candidate_mask"],
                        objective=contract["objective"],
                        brier_weight=contract["brier_weight"],
                        teacher_probs=batch["teacher_probs"],
                        replay_mask=batch["replay_mask"],
                        replay_kl_weight=contract["replay_kl_weight"],
                    )
                    loss = terms["total"].sum() / ACCUMULATION
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite target loss")
                if step == LAST_UPDATE:
                    torch.cuda.synchronize(device)
                    events.emit(
                        step=step, microbatch=microbatch, phase="backward_start"
                    )
                loss.backward()
                if step == LAST_UPDATE:
                    torch.cuda.synchronize(device)
                    events.emit(
                        step=step, microbatch=microbatch, phase="backward_complete"
                    )
                    active["enabled"] = False
                weighted_total += terms["total"].detach().sum().item()
                tokens += batch["attention_mask"].sum().item()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gradient_norm):
                raise RuntimeError("Nonfinite gradient norm")
            optimizer.step()
            torch.cuda.synchronize(device)
            observed = {
                "tokens": tokens,
                "loss": weighted_total / ACCUMULATION,
                "gradient_norm": gradient_norm.item(),
            }
            if step < LAST_UPDATE:
                deviations = compare_step(step, observed, expected_steps[step])
                largest_deviation = {
                    key: max(largest_deviation[key], value)
                    for key, value in deviations.items()
                }
            events.emit(step=step, phase="update_complete", **observed)
        return {
            "status": "EXACT_STATE_WINDOW_PASS",
            "updates_replayed": LAST_UPDATE - FIRST_UPDATE + 1,
            "max_relative_deviation": largest_deviation,
            "events_sha256": sha256(args.events),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "saved_weights_modified": False,
        }
    finally:
        for handle in handles:
            handle.remove()
        events.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--console", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--events", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(continue_state(args), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
