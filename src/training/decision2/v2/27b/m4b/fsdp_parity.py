"""P0 engineering parity for the M4b trainer: sharded FSDP2 vs a 1-process reference.

A tiny random Qwen3.5-architecture model (gated-delta linear-attention and
full-attention layers, vision tower, saved so ``DecisionModel.from_base`` loads
it like the real base; the image's FLA / causal-conv1d kernels on GPU) runs two
planned updates in FP32 by ``train_ff``'s own code path (``build_model``,
``prepare_model``, ``rank_schedule``, ``accumulate_update`` / ``run_update``,
``save_checkpoint``). The second update holds fewer rows than ranks, so one
rank does zero-weight dummy work only. Rank 0 repeats everything on an
unsharded copy, processing every rank's micro-batches in turn (dummies
included), so sharding and reduction are the only differences.

Gradient parity: for each planned update, at the initial parameters, the
gathered FSDP gradient of every tensor against the reference gradient. A tensor
whose reference norm is at least 1e-6 of the global reference norm must be
within 1e-4 relative; any other tensor's difference norm must be at most 1e-6
of the global reference norm; the concatenated gradient must be within 1e-5
relative. Two-update AdamW run: per-update loss and pre-clip gradient norm
within 1e-5 relative, every tensor updated, all finite, and the saved checkpoint
reloaded by ``DecisionModel.from_checkpoint`` bitwise equal to the gathered
FSDP state; on GPU the FLA kernels bound. Parameters after AdamW are reported
only (``report_only``): Adam's per-element normalization neither detects a
gradient scale error nor tolerates reduction-order noise on tiny gradients. A
second reference with the 1-rank schedule is also reported. Writes
``parity.json``; exit 0 on PASS.

Run from ``src/training/decision2``::

    python3 -m torch.distributed.run --nnodes 1 --master-addr 127.0.0.1 --master-port 29500 \\
        --nproc-per-node 3 \\
        -m v2.27b.m4b.fsdp_parity --output DIR
    python3 -m v2.27b.m4b.fsdp_parity --output DIR   # FSDP over one rank
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from training.model.decision_model import DecisionModel, encode
from training.model.train import atomic_json

train_ff = importlib.import_module("v2.27b.m4b.train_ff")
tiny = importlib.import_module("v2.27b.m4b.tiny")

LOSS_TOLERANCE = 1e-5
PARAMETER_TOLERANCE = 1e-4
GRADIENT_TENSOR_TOLERANCE = 1e-4
GRADIENT_GLOBAL_TOLERANCE = 1e-5
GRADIENT_FLOOR = 1e-6
REVISION = "tiny-random"
WORST = 5


def relative(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-12)


def tensor_relative(a: torch.Tensor, b: torch.Tensor) -> float:
    return (
        (a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-12)
    ).item()


def prepared(group: Any, args: argparse.Namespace, source: Path) -> tuple[Any, Any]:
    model, tokenizer = train_ff.build_model(source, REVISION, args.head_dim, args.seed)
    return train_ff.prepare_model(model, group, "fp32", checkpointing=True), tokenizer


def full_gradients(model: torch.nn.Module, group: Any) -> dict[str, torch.Tensor]:
    """Unsharded CPU gradient of every parameter on rank 0 (empty elsewhere); a collective when sharded."""
    out = {}
    for name, parameter in model.named_parameters():
        grad = parameter.grad
        if grad is None:
            grad = torch.zeros(parameter.shape, dtype=parameter.dtype)
        elif hasattr(grad, "full_tensor"):
            grad = grad.full_tensor()
        if group.main:
            out[name] = grad.detach().cpu().clone()
    return out


def update_gradients(
    group: Any,
    args: argparse.Namespace,
    source: Path,
    items: list[dict[str, Any]],
    dummy: dict[str, Any],
    plan: list[list[int]],
    schedules: list[list[list[int] | None]],
    pad_id: int,
) -> list[tuple[list[float], dict[str, torch.Tensor]]]:
    """Loss sums and gradients of every planned update, each at the initial parameters."""
    model, _ = prepared(group, args, source)
    model.train()
    out = []
    for rows, micro_batches in zip(plan, schedules):
        totals = train_ff.accumulate_update(
            model,
            group,
            micro_batches=micro_batches,
            rows=len(rows),
            items=items,
            dummy=dummy,
            pad_id=pad_id,
            precision="fp32",
            brier_weight=0.5,
            kl_weight=0.0,
        )
        out.append((totals, full_gradients(model, group)))
    model.zero_grad(set_to_none=True)
    del model
    return out


def compare_gradients(
    sharded: dict[str, torch.Tensor], reference: dict[str, torch.Tensor]
) -> dict[str, Any]:
    names = sorted(reference)
    if sorted(sharded) != names:
        raise RuntimeError("Sharded and reference gradients name different tensors")
    difference = {
        k: (sharded[k].double() - reference[k].double()).norm().item() for k in names
    }
    norm = {k: reference[k].double().norm().item() for k in names}
    total = math.sqrt(sum(v * v for v in norm.values()))
    spread = math.sqrt(sum(v * v for v in difference.values()))
    measured = {
        k: difference[k] / norm[k] for k in names if norm[k] >= GRADIENT_FLOOR * total
    }
    small = {k: difference[k] / max(total, 1e-30) for k in names if k not in measured}
    worst = sorted(measured, key=measured.get, reverse=True)[:WORST]
    return {
        "reference_norm": total,
        "global_relative": spread / max(total, 1e-30),
        "tensors": len(names),
        "tensors_below_floor": len(small),
        "tensor_relative_max": max(measured.values(), default=0.0),
        "worst_tensors": [
            {"name": k, "relative": measured[k], "reference_norm": norm[k]}
            for k in worst
        ],
        "below_floor_difference_max": max(small.values(), default=0.0),
        "below_floor_worst": max(small, key=small.get) if small else None,
        "passed": {
            "tensors": total > 0
            and all(v <= GRADIENT_TENSOR_TOLERANCE for v in measured.values()),
            "below_floor": all(v <= GRADIENT_FLOOR for v in small.values()),
            "global": spread <= GRADIENT_GLOBAL_TOLERANCE * total,
        },
    }


def train_two(
    group: Any,
    args: argparse.Namespace,
    source: Path,
    items: list[dict[str, Any]],
    dummy: dict[str, Any],
    plan: list[list[int]],
    schedules: list[list[list[int] | None]],
    pad_id: int,
) -> tuple[Any, Any, list[dict[str, Any]]]:
    model, tokenizer = prepared(group, args, source)
    optimizer = train_ff.make_optimizer(model, args.backbone_lr, args.head_lr, 0.01)
    model.train()
    results = []
    for step, (rows, micro_batches) in enumerate(zip(plan, schedules)):
        results.append(
            train_ff.run_update(
                model,
                optimizer,
                group,
                micro_batches=micro_batches,
                rows=len(rows),
                items=items,
                dummy=dummy,
                pad_id=pad_id,
                precision="fp32",
                brier_weight=0.5,
                kl_weight=0.0,
                step=step,
                horizon=len(plan),
                warmup_ratio=0.0,
                clip=1.0,
            )
        )
    return model, tokenizer, results


def compare_runs(
    sharded: list[dict[str, Any]],
    reference: list[dict[str, Any]],
    state: dict[str, torch.Tensor],
    reference_state: dict[str, torch.Tensor],
    initial: dict[str, torch.Tensor],
) -> dict[str, Any]:
    return {
        "loss_relative": [
            relative(a["loss"], b["loss"]) for a, b in zip(sharded, reference)
        ],
        "grad_norm_relative": [
            relative(a["grad_norm_preclip"], b["grad_norm_preclip"])
            for a, b in zip(sharded, reference)
        ],
        "tensors_changed": sum(not torch.equal(state[k], initial[k]) for k in state),
        "tensors": len(state),
        "parameters_after_adamw": compare_parameters(state, reference_state, initial),
    }


def compare_parameters(
    state: dict[str, torch.Tensor],
    reference_state: dict[str, torch.Tensor],
    initial: dict[str, torch.Tensor],
) -> dict[str, Any]:
    per_tensor = {
        name: tensor_relative(state[name], reference_state[name]) for name in state
    }
    nonzero = {k: v for k, v in per_tensor.items() if initial[k].abs().sum() > 0}
    zero_init = {k: v for k, v in per_tensor.items() if k not in nonzero}
    delta = torch.cat(
        [(state[k] - initial[k]).double().flatten() for k in sorted(state)]
    )
    reference_delta = torch.cat(
        [(reference_state[k] - initial[k]).double().flatten() for k in sorted(state)]
    )
    return {
        "parameter_relative_max": max(nonzero.values()),
        "parameter_relative_worst": max(nonzero, key=nonzero.get),
        "zero_init_parameter_relative": zero_init,
        "update_relative": (
            (delta - reference_delta).norm() / reference_delta.norm().clamp_min(1e-30)
        ).item(),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arch", choices=("qwen3_5", "qwen3"), default="qwen3_5")
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--rows", type=int, default=24)
    parser.add_argument("--update-rows", type=int, default=12)
    parser.add_argument("--max-batch-tokens", type=int, default=256)
    parser.add_argument("--max-batch-rows", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--backbone-lr", type=float, default=1e-3)
    parser.add_argument("--head-lr", type=float, default=1e-2)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    group = train_ff.init_group(args.device, 30)
    runtime = None
    if group.device.type == "cuda":
        from v2.dec.runtime_check import require_runtime

        runtime = require_runtime()
    source = args.output / "source"
    if group.main:
        args.output.mkdir(parents=True, exist_ok=True)
        if (args.output / "parity.json").exists() or source.exists():
            raise FileExistsError(f"{args.output} already holds a parity run")
        tiny.write_source(source, arch=args.arch, layers=args.layers, seed=args.seed)
    group.barrier()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    items = [encode(row, tokenizer, 4096) for row in tiny.make_rows(args.rows, "train")]
    lengths = [len(item["ids"]) for item in items]
    plan = train_ff.update_plan(
        lengths,
        seed=args.seed,
        max_tokens=args.max_batch_tokens,
        max_rows=args.max_batch_rows,
        update_rows=args.update_rows,
    )[:2]
    plan[1] = plan[1][: max(1, group.world - 1)]
    dummy = items[min(range(len(lengths)), key=lambda i: (lengths[i], i))]
    pad_id = tokenizer.pad_token_id

    def schedule(world: int) -> list[list[list[int] | None]]:
        return [
            train_ff.rank_schedule(
                rows,
                lengths,
                world,
                max_tokens=args.max_batch_tokens,
                max_rows=args.max_batch_rows,
            )
            for rows in plan
        ]

    sharded_schedule = schedule(group.world)
    own_schedule = [per_rank[group.rank] for per_rank in sharded_schedule]
    reference_schedule = [
        [batch for per_rank in ranks for batch in per_rank]
        for ranks in sharded_schedule
    ]
    sharded_gradients = update_gradients(
        group, args, source, items, dummy, plan, own_schedule, pad_id
    )
    model, tokenizer, sharded = train_two(
        group, args, source, items, dummy, plan, own_schedule, pad_id
    )
    state = train_ff.gather_state(model, group)
    name = train_ff.save_checkpoint(
        model,
        tokenizer,
        group,
        output=args.output / "fsdp",
        step=len(plan),
        metrics=None,
    )
    del model
    if group.main:
        plain = train_ff.Group(0, 1, group.device, sharded=False)
        initial, _ = train_ff.build_model(source, REVISION, args.head_dim, args.seed)
        initial = {k: v.detach().clone() for k, v in initial.state_dict().items()}
        reference_gradients = update_gradients(
            plain, args, source, items, dummy, plan, reference_schedule, pad_id
        )
        gradient_parity = [
            {
                "loss_relative": relative(ours[0][0], theirs[0][0]),
                **compare_gradients(ours[1], theirs[1]),
            }
            for ours, theirs in zip(sharded_gradients, reference_gradients)
        ]
        del reference_gradients
        reference_model, _, reference = train_two(
            plain, args, source, items, dummy, plan, reference_schedule, pad_id
        )
        reference_state = train_ff.gather_state(reference_model, plain)
        del reference_model
        native_model, _, native = train_two(
            plain,
            args,
            source,
            items,
            dummy,
            plan,
            [ranks[0] for ranks in schedule(1)],
            pad_id,
        )
        native_state = train_ff.gather_state(native_model, plain)
        del native_model
        reloaded, _ = DecisionModel.from_checkpoint(args.output / "fsdp" / name)
        reloaded_state = reloaded.state_dict()
        bitwise = set(reloaded_state) == set(state) and all(
            reloaded_state[k].dtype == torch.float32
            and torch.equal(reloaded_state[k], state[k])
            for k in state
        )
        against_reference = compare_runs(
            sharded, reference, state, reference_state, initial
        )
        after_adamw = against_reference.pop("parameters_after_adamw")
        gates = {
            "gradient_tensors_within_1e-4": all(
                update["passed"]["tensors"] for update in gradient_parity
            ),
            "gradient_below_floor_within_1e-6": all(
                update["passed"]["below_floor"] for update in gradient_parity
            ),
            "gradient_global_within_1e-5": all(
                update["passed"]["global"] for update in gradient_parity
            ),
            "loss_within_1e-5": all(
                v <= LOSS_TOLERANCE for v in against_reference["loss_relative"]
            ),
            "grad_norm_within_1e-5": all(
                v <= LOSS_TOLERANCE for v in against_reference["grad_norm_relative"]
            ),
            "every_tensor_updated": against_reference["tensors_changed"]
            == against_reference["tensors"],
            "all_finite": all(
                math.isfinite(result[key])
                for result in sharded
                for key in ("loss", "grad_norm_preclip")
            ),
            "save_reload_bitwise": bitwise,
        }
        if group.device.type == "cuda":
            gates["kernel_runtime"] = runtime is not None
        report = {
            "schema_version": "decision2-27b-m4b-fsdp-parity/2",
            "status": "PASS" if all(gates.values()) else "FAIL",
            "gates": gates,
            "arch": args.arch,
            "world_size": group.world,
            "device": (
                torch.cuda.get_device_name(group.device)
                if group.device.type == "cuda"
                else "cpu"
            ),
            "plan_rows": [len(rows) for rows in plan],
            "micro_batches_per_rank": [
                [sum(1 for b in part if b) for part in ranks]
                for ranks in sharded_schedule
            ],
            "dummy_micro_batches_per_rank": [
                [sum(1 for b in part if not b) for part in ranks]
                for ranks in sharded_schedule
            ],
            "gradient_parity": gradient_parity,
            "sharded": sharded,
            "reference": reference,
            "against_reference": against_reference,
            "report_only_after_adamw": {
                "parameters_within_1e-4": after_adamw["parameter_relative_max"]
                <= PARAMETER_TOLERANCE,
                "updates_within_1e-4": after_adamw["update_relative"]
                <= PARAMETER_TOLERANCE,
                **after_adamw,
            },
            "against_native_schedule_reference": compare_runs(
                sharded, native, state, native_state, initial
            ),
            "checkpoint": str(args.output / "fsdp" / name),
            "tolerances": {
                "loss": LOSS_TOLERANCE,
                "gradient_tensor": GRADIENT_TENSOR_TOLERANCE,
                "gradient_global": GRADIENT_GLOBAL_TOLERANCE,
                "gradient_floor": GRADIENT_FLOOR,
                "parameters_report_only": PARAMETER_TOLERANCE,
            },
            "runtime": runtime,
            "torch_version": torch.__version__,
        }
        atomic_json(args.output / "parity.json", report)
        print(json.dumps({"status": report["status"], "gates": gates}), flush=True)
    group.barrier()
    status = group.gather(
        json.loads((args.output / "parity.json").read_text())["status"]
        if group.main
        else None
    )[0]
    if group.sharded:
        torch.distributed.destroy_process_group()
    raise SystemExit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
